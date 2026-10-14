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
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        # Causal language models shift labels internally to predict the next token.
        batch["labels"] = torch.where(batch["input_ids"] == tokenizer.pad_token_id, -100, batch["input_ids"])

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
    from accelerate.test_utils.training import mocked_dataloaders_for_autoregressive_models

    get_dataloaders = mocked_dataloaders_for_autoregressive_models  # noqa: F811


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

    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    max_grad_norm = config["max_grad_norm"]

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
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M").to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = get_constant_schedule(optimizer=optimizer)

    train(
        accelerator,
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        num_epochs=num_epochs,
        max_grad_norm=max_grad_norm,
        with_wandb_tracking=args.with_wandb_tracking,
    )
    accelerator.end_training()


def train(
    accelerator,
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
    lr_scheduler,
    *,
    num_epochs,
    max_grad_norm,
    with_wandb_tracking=False,
):
    """Prepare the training objects, then train and evaluate for each epoch."""
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_batches_in_epoch = len(train_dataloader)
    num_batches_in_last_update = num_batches_in_epoch % gradient_accumulation_steps
    if num_batches_in_last_update == 0:
        num_batches_in_last_update = gradient_accumulation_steps
    total_gradient_updates = math.ceil(num_batches_in_epoch / gradient_accumulation_steps)

    for epoch in range(num_epochs):
        model.train()
        training_iterator = iter(train_dataloader)
        for update_step in range(total_gradient_updates):
            # Every microbatch needs the target count for the entire update before backward.
            num_batches_in_step = (
                gradient_accumulation_steps
                if update_step != (total_gradient_updates - 1)
                else num_batches_in_last_update
            )
            batches = [next(training_iterator) for _ in range(num_batches_in_step)]

            # The first label has no preceding prediction and is excluded by the model's loss.
            local_num_items_in_batch = sum(batch["labels"][..., 1:].ne(-100).sum() for batch in batches)

            # Include targets from every process in the loss denominator.
            num_items_in_batch = accelerator.gather(local_num_items_in_batch).sum().item()
            losses = []
            for i, batch in enumerate(batches):
                # Avoid repeated communication by synchronizing only on the last microbatch.
                sync_context = (
                    model.no_sync
                    if (i < len(batches) - 1 and accelerator.num_processes > 1)
                    else contextlib.nullcontext
                )
                with sync_context():
                    outputs = model(**batch, use_cache=False, num_items_in_batch=num_items_in_batch)
                    loss = outputs.loss

                    # The model already divides by the global target count. Cancel the
                    # additional averaging from Accelerate's backward and DDP's reduction.
                    loss = loss * gradient_accumulation_steps * accelerator.num_processes
                    accelerator.backward(loss)
                    losses.append(loss.detach())

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
            if with_wandb_tracking:
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
        if with_wandb_tracking:
            accelerator.log(
                {
                    "eval/perplexity": perplexity,
                    "eval/loss": eval_loss,
                    "eval/epoch": epoch,
                },
                step=update_step + total_gradient_updates * epoch,
            )


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
        help="The size of each minibatch",
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
