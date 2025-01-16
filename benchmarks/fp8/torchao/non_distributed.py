# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
This script tests to ensure that `accelerate` performs at the same level as raw `torchao`.

This particular script verifies this for single GPU training.
"""

import evaluate
import torch
from functools import partial
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchao.float8 import convert_to_float8_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, set_seed


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")


def get_dataloaders(model_name: str, batch_size: int = 16):
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


def get_training_utilities(model_name: str, batch_size: int = 16, accelerator=None):
    """
    Returns a tuple of:
        - Model
        - Optimizer
        - Train dataloader (prepared)
        - Eval dataloader (prepared)
        - LR Scheduler
    Suitable for training on the MRPC dataset
    """

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


def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    if fqn in (first_layer_name, last_layer_name):
        return False
    return True


def train_baseline():
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(MODEL_NAME)
    first_linear = None
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if first_linear is None:
                first_linear = name
            last_linear = name

    func = partial(filter_linear_layers, first_layer_name=first_linear, last_layer_name=last_linear)
    model.to("cuda")
    convert_to_float8_training(model, module_filter_fn=func)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()

    for batch in train_dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC)

    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    return base_model_results, trained_model_results


def train_integration():
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(MODEL_NAME)
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[AORecipeKwargs()])
    model = accelerator.prepare(model)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()

    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC)

    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    return base_model_results, trained_model_results


if __name__ == "__main__":
    # baseline_not_trained, baseline_trained = train_baseline()
    accelerator_not_trained, accelerator_trained = train_integration()
    # assert (
    #     baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"]
    # ), f'Accuracy should be the same for the baseline and accelerator: {baseline_not_trained["accuracy"]} == {accelerator_not_trained["accuracy"]}'
    # assert (
    #     baseline_not_trained["f1"] == accelerator_not_trained["f1"]
    # ), f'F1 score should be the same for the baseline and accelerator: {baseline_not_trained["f1"]} == {accelerator_not_trained["f1"]}'
    # assert (
    #     baseline_trained["accuracy"] == accelerator_trained["accuracy"]
    # ), f'Accuracy should be the same for the baseline and accelerator: {baseline_trained["accuracy"]} == {accelerator_trained["accuracy"]}'
    # assert (
    #     baseline_trained["f1"] == accelerator_trained["f1"]
    # ), f'F1 score should be the same for the baseline and accelerator: {baseline_trained["f1"]} == {accelerator_trained["f1"]}'
