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

import math
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import datasets
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import is_tpu_available, set_seed
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_basic_setup(accelerator, num_samples=82):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=num_samples)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    return model, ddp_model, dataloader


def get_dataloader(accelerator: Accelerator, use_longest=False):
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
        if use_longest:
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")
        return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")

    return DataLoader(tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=16)


def get_mrpc_setup(dispatch_batches, split_batches):
    accelerator = Accelerator(dispatch_batches=dispatch_batches, split_batches=split_batches)
    dataloader = get_dataloader(accelerator, not dispatch_batches)
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/mrpc-bert-base-cased", return_dict=True
    )
    ddp_model, ddp_dataloader = accelerator.prepare(model, dataloader)
    return {"ddp": [ddp_model, ddp_dataloader, "cuda:0"], "no": [model, dataloader, accelerator.device]}, accelerator


def generate_predictions(model, dataloader, accelerator):
    logits_and_targets = []
    for batch in dataloader:
        input, target = batch.values()
        with torch.no_grad():
            logit = model(input)
            logit, target = accelerator.gather_for_metrics((logit, target))
            logits_and_targets.append((logit, target))
    logits, targs = [], []
    for (logit, targ) in logits_and_targets:
        logits.append(logit)
        targs.append(targ)
    logits, targs = torch.cat(logits), torch.cat(targs)
    return logits, targs


def test_torch_metrics(accelerator: Accelerator, num_samples=82, dispatch_batches=False, split_batches=False):
    model, ddp_model, dataloader = get_basic_setup(accelerator, num_samples)
    logits, targs = generate_predictions(ddp_model, dataloader, accelerator)
    assert (
        len(logits) == num_samples
    ), f"Unexpected number of inputs:\n    Expected: {num_samples}\n    Actual: {len(logits)}"


def test_mrpc(dispatch_batches: bool = False, split_batches: bool = False):
    metric = evaluate.load("glue", "mrpc")
    setup, accelerator = get_mrpc_setup(dispatch_batches, split_batches)
    # First do baseline
    model, dataloader, device = setup["no"]
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
        assert math.isclose(
            baseline[key], distributed[key]
        ), f"Baseline and Distributed are not the same for key {key}:\n\tBaseline: {baseline[key]}\n\tDistributed: {distributed[key]}\n"


def main():
    accelerator = Accelerator(split_batches=False, dispatch_batches=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # These are a bit slower so they should only be ran on the GPU or TPU
    if torch.cuda.is_available() or is_tpu_available():
        if accelerator.is_local_main_process:
            print("**Testing gather_for_metrics**")
        for split_batches in [True, False]:
            for dispatch_batches in [True, False]:
                if accelerator.is_local_main_process:
                    print(f"With: `split_batches={split_batches}`, `dispatch_batches={dispatch_batches}`")
                test_mrpc(dispatch_batches, split_batches)
                accelerator.state._reset_state()
    if accelerator.is_local_main_process:
        print("**Test torch metrics**")
    for split_batches in [True, False]:
        for dispatch_batches in [True, False]:
            accelerator = Accelerator(split_batches=split_batches, dispatch_batches=dispatch_batches)
            if accelerator.is_local_main_process:
                print(f"With: `split_batches={split_batches}`, `dispatch_batches={dispatch_batches}`, length=99")
            test_torch_metrics(accelerator, 99)
            accelerator.state._reset_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
