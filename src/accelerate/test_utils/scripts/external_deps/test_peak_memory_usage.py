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
import gc
import json
import os

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from accelerate.utils import is_mlu_available, is_musa_available, is_npu_available, is_xpu_available
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.cuda.memory_allocated()
        elif is_mlu_available():
            torch.mlu.empty_cache()
            torch.mlu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.mlu.memory_allocated()
        elif is_musa_available():
            torch.musa.empty_cache()
            torch.musa.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.musa.memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            torch.npu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.npu.memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.xpu.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
        elif is_mlu_available():
            torch.mlu.empty_cache()
            torch.mlu.memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.mlu.max_memory_allocated()
        elif is_musa_available():
            torch.musa.empty_cache()
            torch.musa.memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.musa.max_memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            self.end = torch.npu.memory_allocated()
            self.peak = torch.npu.max_memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            self.end = torch.xpu.memory_allocated()
            self.peak = torch.xpu.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def get_dataloaders(
    accelerator: Accelerator,
    batch_size: int = 16,
    model_name: str = "bert-base-cased",
    n_train: int = 320,
    n_val: int = 160,
):
    """
    Creates a set of `DataLoader`s for the `glue` dataset.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
        model_name (`str`, *optional*):
            The name of the model to use.
        n_train (`int`, *optional*):
            The number of training examples to use.
        n_val (`int`, *optional*):
            The number of validation examples to use.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = load_dataset(
        "glue", "mrpc", split={"train": f"train[:{n_train}]", "validation": f"validation[:{n_val}]"}
    )

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"], load_from_cache_file=False
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.XLA:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator()

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    model_name = args.model_name_or_path

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size, model_name, args.n_train, args.n_val)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)

    # Instantiate optimizer
    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=lr)

    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    else:
        gradient_accumulation_steps = 1
    max_training_steps = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps

    # Instantiate scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(optimizer, total_num_steps=max_training_steps, warmup_num_steps=0)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Now we train the model
    train_total_peak_memory = {}
    for epoch in range(starting_epoch, num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                overall_step += 1

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(f"Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )
        train_total_peak_memory[f"epoch-{epoch}"] = tracemalloc.peaked + b2mb(tracemalloc.begin)
        if args.peak_memory_upper_bound is not None:
            assert (
                train_total_peak_memory[f"epoch-{epoch}"] <= args.peak_memory_upper_bound
            ), "Peak memory usage exceeded the upper bound"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "peak_memory_utilization.json"), "w") as f:
            json.dump(train_total_peak_memory, f)
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script tracking peak GPU memory usage.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-cased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--peak_memory_upper_bound",
        type=float,
        default=None,
        help="The upper bound of peak memory usage in MB. If set, the training will throw an error if the peak memory usage exceeds this value.",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=320,
        help="Number of training examples to use.",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=160,
        help="Number of validation examples to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of train epochs.",
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": args.num_epochs, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
