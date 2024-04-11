# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import csv

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
from transformers.data.data_collator import DataCollatorWithPadding


model_name = "distilbert-base-uncased"
dataset_name = "imdb"
metric = "accuracy"

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    datasets = load_dataset("imdb")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
        )

    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=32,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    from accelerate.utils import FP8RecipeKwargs
    recipe = FP8RecipeKwargs()
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, kwargs_handlers=[recipe])
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    metric = evaluate.load("accuracy")

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id, return_dict=True
    )
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    accelerator.print(f"State: {accelerator.state}")
    training_data = {"step": [], "loss": []}
    valid_data = {"step": [], "loss": [], "accuracy": []}
    current_step = 0


    # Now we train the model
    epoch_pbar = tqdm(True, range(num_epochs))
    for epoch in epoch_pbar:
        model.train()
        epoch_pbar.set_description(f"Epoch {epoch}")
        train_pbar = tqdm(True, train_dataloader)
        for batch in train_pbar:
            outputs = model(**batch)
            loss = outputs.loss
            training_data["step"].append(current_step)
            training_data["loss"].append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_step += 1

        model.eval()
        eval_pbar = tqdm(True, eval_dataloader)
        for batch in eval_pbar:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            valid_data["step"].append(current_step)
            valid_data["loss"].append(loss.item())
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        valid_data["accuracy"].append(eval_metric)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
    with open(f"training_results_{accelerator.mixed_precision}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(training_data.keys())
        writer.writerows(zip(*training_data.values()))
    with open(f"valid_results_{accelerator.mixed_precision}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(valid_data.keys())
        writer.writerows(zip(*valid_data.values()))




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
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()