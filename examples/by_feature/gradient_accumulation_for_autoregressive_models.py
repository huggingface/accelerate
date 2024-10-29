# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import contextlib
import math
import os

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule, set_seed

from accelerate import Accelerator, DistributedType


########################################################################
# This is a fully working simple example to use Accelerate
# and perform gradient accumulation on samples of variable size
#
# This example trains a SmolLM base model on WikiText-2 v1
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


EVAL_BATCH_SIZE = 32


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16, max_training_samples=500):
    """
    Creates a set of `DataLoader`s for the `Salesforce/wikitext` dataset,
    using "HuggingFaceTB/SmolLM-360M" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
    tokenizer.pad_token = tokenizer.eos_token
    with accelerator.local_main_process_first():
        datasets = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
        datasets["train"] = datasets["train"].select(range(max_training_samples))

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["text"], truncation=True, max_length=None, return_attention_mask=False)
        return outputs

    # Filter out empty texts
    with accelerator.main_process_first():
        datasets = datasets.filter(
            lambda x: len(x) > 0,
            input_columns="text",
        )

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

    # Filter out empty samples
    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.filter(
            lambda x: len(x) > 0,
            input_columns="input_ids",
        )

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = (
            128
            if accelerator.distributed_type == DistributedType.XLA
            else max([len(e["input_ids"]) for e in examples])
        )
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        batch = tokenizer.pad(
            examples,
            padding="max_length",
            max_length=max_length + 1,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = batch["input_ids"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]

        batch["labels"] = torch.where(batch["labels"] == tokenizer.pad_token_id, -100, batch["labels"])

        return batch

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader


# For testing only
if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
    from accelerate.test_utils.training import mocked_dataloaders

    get_dataloaders = mocked_dataloaders  # noqa: F811


def training_function(config, args):
    # For testing only
    if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
        config["num_epochs"] = 2

    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    # Initialize accelerator
    if args.with_wandb_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb",
        )
    else:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps
        )
    if accelerator.distributed_type == DistributedType.XLA and gradient_accumulation_steps > 1:
        raise NotImplementedError(
            "Gradient accumulation on TPUs is currently not supported. Pass `gradient_accumulation_steps=1`"
        )
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    max_grad_norm = config["max_grad_norm"]

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_wandb_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        run_name = f"{accelerator.num_processes}GPU-grad{gradient_accumulation_steps}-bs{batch_size}"
        accelerator.init_trackers(
            run,
            config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_constant_schedule(
        optimizer=optimizer,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_samples_in_epoch = len(train_dataloader)
    remainder = num_samples_in_epoch % gradient_accumulation_steps
    remainder = remainder if remainder != 0 else gradient_accumulation_steps
    total_gradient_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)

    total_batched_samples = 0
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        training_iterator = iter(train_dataloader)
        for update_step in range(total_gradient_updates):
            # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
            # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
            batch_samples = []
            num_batches_in_step = (
                gradient_accumulation_steps if update_step != (total_gradient_updates - 1) else remainder
            )
            for _ in range(num_batches_in_step):
                batch_samples += [next(training_iterator)]
            # get local num items in batch
            local_num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])

            # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
            num_items_in_batch = accelerator.gather(local_num_items_in_batch).sum().item()
            losses = []
            for i, batch in enumerate(batch_samples):
                # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unecessary communications when accumulating
                # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
                ctx = (
                    model.no_sync
                    if (i < len(batch_samples) - 1 and accelerator.num_processes > 1)
                    else contextlib.nullcontext
                )
                with ctx():
                    total_batched_samples += 1

                    outputs = model(**batch, use_cache=False, num_items_in_batch=num_items_in_batch)
                    loss = outputs.loss

                    # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                    # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps
                    # Because the loss is already divided by `num_items_in_batch` in the `transformers` code, we don't need to do it again
                    loss = loss * gradient_accumulation_steps * accelerator.num_processes
                    accelerator.backward(loss)
                    losses.append(loss.detach())

            # Sync gradients and perform optimization steps once every gradient_accumulation_steps
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            losses = accelerator.gather(sum(losses)).sum().item() / (
                accelerator.num_processes * gradient_accumulation_steps
            )

            grad_norm = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            accelerator.print(
                f"epoch {epoch} - update step {update_step}:: grad norm: {grad_norm} ::train loss: {losses}"
            )
            if args.with_wandb_tracking:
                accelerator.log(
                    {
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "train/loss": losses,
                    },
                    step=update_step + total_gradient_updates * epoch,
                )
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch, use_cache=False)
            eval_loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(EVAL_BATCH_SIZE)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:: eval perplexity: {perplexity} eval_loss: {eval_loss}")
        if args.with_wandb_tracking:
            accelerator.log(
                {
                    "eval/perplexity": perplexity,
                    "eval/loss": eval_loss,
                    "eval/epoch": epoch,
                },
                step=update_step + total_gradient_updates * epoch,
            )
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The number of minibatches to be ran before gradients are accumulated.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=2,
        help="The number of minibatches to be ran before gradients are accumulated.",
    )

    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--with_wandb_tracking",
        action="store_true",
        help="Whether to load in wandb from the environment and use them for logging.",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": args.per_device_batch_size, "max_grad_norm": 1.0}
    training_function(config, args)


if __name__ == "__main__":
    main()
