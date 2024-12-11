# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate.utils.dataclasses import DistributedType


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=None):
        rng = np.random.default_rng(seed)
        self.length = length
        self.x = rng.normal(size=(length,)).astype(np.float32)
        self.y = a * self.x + b + rng.normal(scale=0.1, size=(length,)).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class RegressionModel4XPU(torch.nn.Module):
    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.b = torch.nn.Parameter(torch.tensor([2, 3]).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a[0] + self.b[0]


class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b


def mocked_dataloaders(accelerator, batch_size: int = 16):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    data_files = {"train": "tests/test_samples/MRPC/train.csv", "validation": "tests/test_samples/MRPC/dev.csv"}
    datasets = load_dataset("csv", data_files=data_files)
    label_list = datasets["train"].unique("label")

    label_to_id = {v: i for i, v in enumerate(label_list)}

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"], examples["sentence2"], truncation=True, max_length=None, padding="max_length"
        )
        if "label" in examples:
            outputs["labels"] = [label_to_id[l] for l in examples["label"]]
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence1", "sentence2", "label"],
    )

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.XLA:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=2)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=1)

    return train_dataloader, eval_dataloader


def mocked_dataloaders_for_autoregressive_models(accelerator, batch_size: int = 16):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
    tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": "tests/test_samples/MRPC/train.csv", "validation": "tests/test_samples/MRPC/dev.csv"}
    datasets = load_dataset("csv", data_files=data_files)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], truncation=True, max_length=None, return_attention_mask=False)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence1", "sentence2", "label"],
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
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=2)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=1)

    return train_dataloader, eval_dataloader
