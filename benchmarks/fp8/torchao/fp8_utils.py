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
import torch


def get_dataloaders(model_name: str, batch_size: int = 16):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(
            examples,
            padding="longest",
            pad_to_multiple_of=16,  # Specific for FP8
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=16,
        drop_last=True,
    )

    return train_dataloader, eval_dataloader


def get_training_utilities(model_name: str, batch_size: int = 16, accelerator=None, prepare=True):
    """
    Returns a tuple of:
        - Model
        - Optimizer
        - Train dataloader (prepared)
        - Eval dataloader (prepared)
        - LR Scheduler
    Suitable for training on the MRPC dataset
    """
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    from accelerate import Accelerator

    if accelerator is None:
        accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    train_dataloader, eval_dataloader = get_dataloaders(model_name, batch_size)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * 2,
    )
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)
    return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler


def get_named_parameters(model):
    """
    Same thing as `Accelerator.get_named_parameters` Returns a list of the named parameters of the model (extracted
    from parallel)
    """
    from accelerate.utils import extract_model_from_parallel

    model = extract_model_from_parallel(model)
    return {n: p for n, p in model.named_parameters()}


def evaluate_model(model, dataloader, metric, accelerator=None):
    "Turns model to .eval(), runs dataloader, calculates metric, then turns eval back on"
    model.eval()
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        if accelerator is not None and accelerator.num_processes > 1:
            predictions, references = accelerator.gather_for_metrics((predictions, references))
        metric.add_batch(predictions=predictions, references=references)
    return metric.compute()
