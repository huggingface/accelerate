# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16, model_name: str = "bert-base-cased"):
    """
    Creates a set of `DataLoader`s for the `glue` dataset.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
        model_name (`str`, *optional*):
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"], load_from_cache_file=False
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.XLA:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader


def evaluation_loop(accelerator, model, eval_dataloader, metric):
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # It is slightly faster to call this once, than multiple times
        predictions, references = accelerator.gather(
            (predictions, batch["labels"])
        )  # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.use_distributed:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    return eval_metric["accuracy"]


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator()

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    model_name = args.model_name_or_path

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size, model_name)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)

    # Instantiate optimizer
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=lr)

    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    else:
        gradient_accumulation_steps = 1
    max_training_steps = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps

    # Instantiate scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(optimizer, total_num_steps=max_training_steps, warmup_num_steps=0)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0
    metric = evaluate.load("glue", "mrpc")
    ending_epoch = num_epochs

    if args.partial_train_epoch is not None:
        ending_epoch = args.partial_train_epoch

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        epoch_string = args.resume_from_checkpoint.split("epoch_")[1]
        state_epoch_num = ""
        for char in epoch_string:
            if char.isdigit():
                state_epoch_num += char
            else:
                break
        starting_epoch = int(state_epoch_num) + 1
        accuracy = evaluation_loop(accelerator, model, eval_dataloader, metric)
        accelerator.print("resumed checkpoint performance:", accuracy)
        accelerator.print("resumed checkpoint's scheduler's lr:", lr_scheduler.get_lr()[0])
        accelerator.print("resumed optimizers's lr:", optimizer.param_groups[0]["lr"])
        with open(os.path.join(args.output_dir, f"state_{starting_epoch - 1}.json")) as f:
            resumed_state = json.load(f)
            assert resumed_state["accuracy"] == accuracy, "Accuracy mismatch, loading from checkpoint failed"
            assert resumed_state["lr"] == lr_scheduler.get_lr()[0], (
                "Scheduler learning rate mismatch, loading from checkpoint failed"
            )
            assert resumed_state["optimizer_lr"] == optimizer.param_groups[0]["lr"], (
                "Optimizer learning rate mismatch, loading from checkpoint failed"
            )
            assert resumed_state["epoch"] == starting_epoch - 1, "Epoch mismatch, loading from checkpoint failed"
            return

    # Now we train the model
    state = {}
    for epoch in range(starting_epoch, ending_epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            overall_step += 1
        output_dir = f"epoch_{epoch}"
        output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)
        accuracy = evaluation_loop(accelerator, model, eval_dataloader, metric)
        state["accuracy"] = accuracy
        state["lr"] = lr_scheduler.get_lr()[0]
        state["optimizer_lr"] = optimizer.param_groups[0]["lr"]
        state["epoch"] = epoch
        state["step"] = overall_step
        accelerator.print(f"epoch {epoch}:", state)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, f"state_{epoch}.json"), "w") as f:
                json.dump(state, f)
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script tracking peak GPU memory usage.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-cased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--partial_train_epoch",
        type=int,
        default=None,
        help="If passed, the training will stop after this number of epochs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of train epochs.",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": args.num_epochs, "seed": 42, "batch_size": 16}

    training_function(config, args)


if __name__ == "__main__":
    main()
