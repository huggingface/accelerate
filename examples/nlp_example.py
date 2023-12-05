# coding=utf-8
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

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
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


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

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
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
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
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

    model.to(accelerator.device)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    from accelerate.utils import convert_bytes

    def print_memory(text, accelerator):
        accelerator.print(f'Memory used {text}: {convert_bytes(torch.cuda.max_memory_allocated())}')
        torch.cuda.reset_peak_memory_stats()
    
    print_memory('before .prepare()', accelerator)
    from torch import nn
    def nested_children(m: torch.nn.Module):
        children = dict(m.named_children())
        output = {}
        if children == {}:
            # if module has no children; m is last child! :O
            return m
        else:
            # look for children from children... to the last child!
            for name, child in children.items():
                try:
                    output[name] = nested_children(child)
                except TypeError:
                    output[name] = nested_children(child)
        return output
    
    children = nested_children(model)
    
    from transformer_engine.pytorch.module.layernorm import LayerNorm as te_LayerNorm
    import transformer_engine.pytorch as te

    def replace_layers(model, keys, replace_func):
        current_module = model

        for key in keys[:-1]:
            if isinstance(key, int):
                current_module = current_module[key]
            elif isinstance(current_module, nn.Module) and key in current_module._modules:
                current_module = current_module._modules[key]
            elif hasattr(current_module, key):
                current_module = getattr(current_module, key)
            else:
                raise KeyError(f"Key '{key}' not found in the model.")

        last_key = keys[-1]
        if isinstance(last_key, int):
            current_module[last_key] = replace_func(current_module[last_key])
        elif isinstance(current_module, nn.Module) and last_key in current_module._modules:
            current_module._modules[last_key] = replace_func(current_module._modules[last_key])
        elif hasattr(current_module, last_key):
            setattr(current_module, last_key, replace_func(getattr(current_module, last_key)))
        else:
            raise KeyError(f"Key '{last_key}' not found in the model.")

        return model
    
    def check_for_layer(dictionary, layer, current_key=None):
        if current_key is None:
            current_key = []
        matching_keys = []

        for k, v in dictionary.items():
            if isinstance(v, layer):
                matching_keys.append(".".join(current_key + [k]))
            elif isinstance(v, dict):
                matching_keys.extend(check_for_layer(v, layer, current_key + [k]))
    
        return matching_keys
    
    # First LayerNorm
    def replace_layernorm(module):
        te_module = te_LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
        module.weight.copy_(te_module.weight)
        module.bias.copy_(te_module.bias)
        return te_module
    layernorm_locations = check_for_layer(children, nn.LayerNorm)
    # then nn.Linear
    def replace_linear(module):
        # Return early if the linear layer weights are not multiples of 16
        if any(p % 16 != 0 for p in module.weight.shape):
            return module
        has_bias = module.bias is not None
        te_module = te.Linear(
            module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
        )
        module.weight.copy_(te_module.weight)
        if has_bias:
            module.bias.copy_(te_module.bias)
        return te_module
    linear_locations = check_for_layer(children, nn.Linear)
    
    with torch.no_grad():
        for location in layernorm_locations:
            model = replace_layers(model, location.split('.'), replace_func=replace_layernorm)

        for location in linear_locations:
            model = replace_layers(model, location.split('.'), replace_func=replace_linear)
    optimizer = AdamW(params=model.parameters(), lr=lr) 

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )
    import msamp

    model, optimizer = msamp.initialize(model, optimizer, opt_level="O2")
    print_memory('after prepare', accelerator)

    
    # Now we train the model
    model.train()
    import time
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        outputs = model(**batch)
        print_memory('after outputs generated', accelerator)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        loss.backward()
        print_memory('after backward', accelerator)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            print_memory('after step', accelerator)
            optimizer.zero_grad()
        if step == 5:
            break
    end_time = time.time()
    print(f'Time taken: {end_time - start_time}')

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
