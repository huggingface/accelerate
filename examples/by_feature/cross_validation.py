# coding=utf-8
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
from typing import List

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset

# New Code #
# We'll be using StratifiedKFold for this example
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType


########################################################################
# This is a fully working simple example to use Accelerate,
# specifically showcasing how to perform Cross Validation,
# and builds off the `nlp_example.py` script.
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To help focus on the differences in the code, building `DataLoaders`
# was refactored into its own function.
# New additions from the base script can be found quickly by
# looking for the # New Code # tags
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

# New Code #
# We need a different `get_dataloaders` function that will build dataloaders by index


def get_fold_dataloaders(
    accelerator: Accelerator, dataset: DatasetDict, train_idxs: List[int], valid_idxs: List[int], batch_size: int = 16
):
    """
    Gets a set of train, valid, and test dataloaders for a particular fold

    Args:
        accelerator (`Accelerator`):
            The main `Accelerator` object
        train_idxs (list of `int`):
            The split indices for the training dataset
        valid_idxs (list of `int`):
            The split indices for the validation dataset
        batch_size (`int`):
            The size of the minibatch. Default is 16
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = DatasetDict(
        {
            "train": dataset["train"].select(train_idxs),
            "validation": dataset["train"].select(valid_idxs),
            "test": dataset["validation"],
        }
    )

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader, test_dataloader


def training_function(config, args):
    # New Code #
    test_predictions = []
    # Download the dataset
    datasets = load_dataset("glue", "mrpc")
    # Create our splits
    kfold = StratifiedKFold(n_splits=int(args.num_folds))
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)

    # New Code #
    # Create our folds:
    folds = kfold.split(np.zeros(datasets["train"].num_rows), datasets["train"]["label"])
    test_references = []
    # Iterate over them
    for i, (train_idxs, valid_idxs) in enumerate(folds):
        train_dataloader, eval_dataloader, test_dataloader = get_fold_dataloaders(
            accelerator,
            datasets,
            train_idxs,
            valid_idxs,
        )
        # Instantiate the model (we build the model here so that the seed also control new weights initialization)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

        # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
        # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
        # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
        model = model.to(accelerator.device)

        # Instantiate optimizer
        optimizer = AdamW(params=model.parameters(), lr=lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
        )

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # Now we train the model
        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}:", eval_metric)

        # New Code #
        # We also run predictions on the test set at the very end
        fold_predictions = []
        for step, batch in enumerate(test_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            fold_predictions.append(predictions.cpu())
            if i == 0:
                # We need all of the test predictions
                test_references.append(references.cpu())
        # Use accelerator.print to print only on the main process.
        test_predictions.append(torch.cat(fold_predictions, dim=0))
        # We now need to release all our memory and get rid of the current model, optimizer, etc
        accelerator.free_memory()
    # New Code #
    # Finally we check the accuracy of our folded results:
    test_references = torch.cat(test_references, dim=0)
    preds = torch.stack(test_predictions, dim=0).sum(dim=0).div(int(args.num_folds)).argmax(dim=-1)
    test_metric = metric.compute(predictions=preds, references=test_references)
    accelerator.print("Average test metrics from all folds:", test_metric)


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
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    # New Code #
    parser.add_argument("--num_folds", type=int, default=3, help="The number of splits to perform across the dataset")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
