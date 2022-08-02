# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import set_seed


def get_setup(accelerator, num_samples=82):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=num_samples)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    return model, ddp_model, dataloader


def generate_predictions(model, dataloader, accelerator):
    logits_and_targets = []
    for batch in dataloader:
        input, target = batch.values()
        with torch.no_grad():
            logits = model(input)
            logits, target = accelerator.gather_for_metrics((logits, target))
            logits_and_targets.append((logits, target))
    inps, targs = [], []
    for (inp, targ) in logits_and_targets:
        inps.append(inp)
        targs.append(targ)
    inps, targs = torch.cat(inps), torch.cat(targs)
    return inps, targs


def test_torch_metrics(accelerator: Accelerator, num_samples=82):
    model, ddp_model, dataloader = get_setup(accelerator, num_samples)
    inps, targs = generate_predictions(ddp_model, dataloader, accelerator)
    assert (
        len(inps) == num_samples
    ), f"Unexpected number of inputs:\n    Expected: {num_samples}\n    Actual: {len(inps)}"


import math

import torch
from torch.utils.data import DataLoader

import evaluate
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_dataloader(accelerator: Accelerator, drop_last=False):
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/mrpc-bert-base-cased")
    dataset = load_dataset("glue", "mrpc", split="validation")

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    return DataLoader(tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=16, drop_last=drop_last)


def get_setup(dispatch_batches, split_batches, drop_last):
    accelerator = Accelerator(dispatch_batches=dispatch_batches, split_batches=split_batches)
    dataloader = get_dataloader(accelerator, drop_last)
    model = AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/mrpc-bert-base-cased", return_dict=True)
    ddp_model, ddp_dataloader = accelerator.prepare(model, dataloader)
    return {"ddp": [ddp_model, ddp_dataloader, "cuda:0"], "no": [model, dataloader, accelerator.device]}, accelerator


def test_mrpc(dispatch_batches: bool = False, split_batches: bool = False):
    drop_last = False if not dispatch_batches else True
    metric = evaluate.load("glue", "mrpc")
    setup, accelerator = get_setup(dispatch_batches, split_batches, drop_last)
    # First do baseline
    if accelerator.is_local_main_process:
        print("Running baseline")
    model, dataloader, device = setup["no"]
    if accelerator.is_local_main_process:
        print(f"Len dl: {len(dataloader)}\nLen dset: {len(dataloader.dataset)}\n")
    model.to(device)
    model.eval()
    for batch in dataloader:
        batch.to(device)
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    baseline = metric.compute()

    # Then do distributed
    if accelerator.is_local_main_process:
        print("Running with Gradient State")
    model, dataloader, device = setup["ddp"]
    model.eval()
    for batch in dataloader:
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        preds, references = accelerator.gather_for_metrics((preds, references))
        metric.add_batch(predictions=preds, references=references)
    distributed = metric.compute()

    for key in "accuracy f1".split():
        if not math.isclose(baseline[key], distributed[key]) and accelerator.is_local_main_process:
            print(
                f"Baseline and Distributed are not the same for key {key}:\n\tBaseline: {baseline[key]}\n\tDistributed: {distributed[key]}\n"
            )



def main():
    accelerator = Accelerator(split_batches=False, dispatch_batches=False)
    if accelerator.is_local_main_process:
        print("**Testing gather_for_metrics**")
    for split_batches in [True, False]:
        for dispatch_batches in [True, False]:
            if accelerator.is_local_main_process:
                print(f"With: `split_batches={split_batches}`, `dispatch_batches={dispatch_batches}`")
            test_mrpc(split_batches, dispatch_batches)
            accelerator.state._reset_state()
    if accelerator.is_local_main_process:
        print("**Test torch metrics**")
    for split_batches in [True, False]:
        for dispatch_batches in [True, False]:
            accelerator = Accelerator(split_batches=split_batches, dispatch_batches=dispatch_batches)
            if accelerator.is_local_main_process:
                print(f"With: `split_batches={split_batches}`, `dispatch_batches={dispatch_batches}`")
            test_torch_metrics(accelerator)
            accelerator.state._reset_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
