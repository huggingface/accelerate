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
from pathlib import Path

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

    max_training_steps = len(train_dataloader) * num_epochs

    # Instantiate scheduler
    linear_decay_scheduler = False
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_training_steps,
        )
        linear_decay_scheduler = True
    else:
        lr_scheduler = DummyScheduler(optimizer, total_num_steps=max_training_steps, warmup_num_steps=0)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Now we train the model
    metric = evaluate.load("glue", "mrpc")
    best_performance = 0
    performance_metric = {}
    expected_lr_after_first_optim_step = lr * (
        1 - 1 / (max_training_steps / accelerator.num_processes / accelerator.gradient_accumulation_steps)
    )
    lr_scheduler_check_completed = False
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # assert the learning rate after first optimizer step
                if (
                    accelerator.sync_gradients
                    and not lr_scheduler_check_completed
                    and linear_decay_scheduler
                    and accelerator.state.mixed_precision == "no"
                ):
                    assert (
                        lr_scheduler.get_last_lr()[0] == expected_lr_after_first_optim_step
                    ), f"Wrong lr found at second step, expected {expected_lr_after_first_optim_step}, got {lr_scheduler.get_last_lr()[0]}"
                    lr_scheduler_check_completed = True

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
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        performance_metric[f"epoch-{epoch}"] = eval_metric["accuracy"]

        if best_performance < eval_metric["accuracy"]:
            best_performance = eval_metric["accuracy"]

    # check that the LR is 0
    if linear_decay_scheduler and accelerator.state.mixed_precision == "no":
        assert (
            lr_scheduler.get_last_lr()[0] == 0
        ), f"Wrong lr found at last step, expected 0, got {lr_scheduler.get_last_lr()[0]}"

    if args.performance_lower_bound is not None:
        assert (
            args.performance_lower_bound <= best_performance
        ), f"Best performance metric {best_performance} is lower than the lower bound {args.performance_lower_bound}"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(performance_metric, f)

    # Finally try saving the model
    accelerator.save_model(model, args.output_dir)
    accelerator.wait_for_everyone()
    assert Path(
        args.output_dir, "model.safetensors"
    ).exists(), "Model was not saved when calling `Accelerator.save_model`"
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
        "--performance_lower_bound",
        type=float,
        default=None,
        help="Optional lower bound for the performance metric. If set, the training will throw error when the performance metric drops below this value.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of train epochs.",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": args.num_epochs, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
