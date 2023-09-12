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

import logging
import math
import os
from copy import deepcopy

import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from accelerate import Accelerator
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import is_tpu_available, set_seed


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class ListHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super(ListHandler, self).__init__(*args, **kwargs)
        self.logs = []

    def emit(self, record):
        self.logs.append(record)


def get_basic_setup(accelerator, num_samples=82, batch_size=16):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=num_samples)
    dataloader = DataLoader(dset, batch_size=batch_size)
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
    for logit, targ in logits_and_targets:
        logits.append(logit)
        targs.append(targ)
    logits, targs = torch.cat(logits), torch.cat(targs)
    return logits, targs


def test_torch_metrics(
    accelerator: Accelerator, num_samples=82, dispatch_batches=False, split_batches=False, batch_size=16
):
    model, ddp_model, dataloader = get_basic_setup(accelerator, num_samples, batch_size)
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


def test_gather_for_metrics_with_non_tensor_objects_iterable_dataset():
    class DummyIterableDataset(IterableDataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for element in self.data:
                yield element

    iterable_dataset = DummyIterableDataset([n for n in range(30)])
    dataloader = DataLoader(iterable_dataset, batch_size=4)
    accelerator = Accelerator()
    prepared_dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        logger = logging.root.manager.loggerDict["accelerate.accelerator"]
        list_handler = ListHandler()
        logger.addHandler(list_handler)

    batches_for_metrics = []
    for batch in prepared_dataloader:
        batches_for_metrics.append(accelerator.gather_for_metrics(batch))

    assert torch.cat(batches_for_metrics).size(0) == 30

    if accelerator.is_main_process:
        assert len(list_handler.logs) == 0
        logger.removeHandler(list_handler)


def test_gather_for_metrics_with_iterable_dataset():
    class DummyIterableDataset(IterableDataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for element in self.data:
                yield element

    iterable_dataset = DummyIterableDataset(torch.as_tensor(range(30)))
    dataloader = DataLoader(iterable_dataset, batch_size=4)

    accelerator = Accelerator()
    prepared_dataloader = accelerator.prepare(dataloader)

    assert isinstance(prepared_dataloader, DataLoaderDispatcher)

    if accelerator.is_main_process:
        logger = logging.root.manager.loggerDict["accelerate.accelerator"]
        list_handler = ListHandler()
        logger.addHandler(list_handler)

    batches_for_metrics = []
    for batch in prepared_dataloader:
        batches_for_metrics.append(accelerator.gather_for_metrics(batch))

    assert torch.cat(batches_for_metrics).size(0) == 30

    if accelerator.is_main_process:
        assert len(list_handler.logs) == 0

        logger.removeHandler(list_handler)


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
        print("test_gather_for_metrics_with_iterable_dataset")
        test_gather_for_metrics_with_iterable_dataset()
        print("test gather_for_metrics_with_non_tensor_objects_iterable_dataset")
        test_gather_for_metrics_with_non_tensor_objects_iterable_dataset()
    if accelerator.is_local_main_process:
        print("**Test torch metrics**")
    for split_batches in [True, False]:
        for dispatch_batches in [True, False]:
            accelerator = Accelerator(split_batches=split_batches, dispatch_batches=dispatch_batches)
            if accelerator.is_local_main_process:
                print(f"With: `split_batches={split_batches}`, `dispatch_batches={dispatch_batches}`, length=99")
            test_torch_metrics(accelerator, 99)
            accelerator.state._reset_state()
    if accelerator.is_local_main_process:
        print("**Test last batch is not dropped when perfectly divisible**")
    accelerator = Accelerator()
    test_torch_metrics(accelerator, 512)
    accelerator.state._reset_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
