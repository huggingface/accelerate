# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from types import MethodType
from typing import Union

import torch
from datasets import load_dataset
from measure_utils import MemoryTracker
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import convert_outputs_to_fp32, set_seed


SEED = 421


def get_named_parameters(model: torch.nn.Module, drop_refs: bool = False) -> dict[str, Union[torch.Tensor, int]]:
    """
    This function returns a dictionary mapping the parameter names to their data pointers or
    the original parameters if `drop_refs` is `False`.
    It is used to get the original parameter names before `fully_shard` is applied.

    We only return the data pointers, so we drop the references to the original parameters
    and `fully_shard` will then trigger a new allocation for the sharded ones.

    Args:
        model (`torch.nn.Module`): Model instance to get the named parameters from
        drop_refs (`bool`, *optional*, defaults to `False`): Whether to drop the references to the original parameters

    Returns:
        `dict[str, Union[torch.Tensor, int]]`: Dictionary mapping the parameter names to their data pointers or the original parameters if `drop_refs` is `False`
    """
    named_parameters = {}
    for n, p in model.named_parameters():
        # We only preserve the data pointers to have the unique 1:1 mapping between the original and the sharded parameters
        named_parameters[n] = p.data_ptr() if drop_refs else p
    return named_parameters


def replace_optimizer_params(optimizer: torch.optim.Optimizer):
    """
    This function is called before using `fully_shard` on the model. It replaces the parameters of the optimizer with
    empty tensors, so `fully_shard` can trigger a new allocation for the sharded ones. After this, we swap the parameters
    `data_ptr` to the original one, so we can reuse that later to map the sharded parameters to the original ones.
    This function modifies the optimizer in-place.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance which contains the original model parameters
    """

    for param_group in optimizer.param_groups:
        for i, p in enumerate(param_group["params"]):
            # We drop a reference to the original param here, so that _move_states_to_device triggers a reallocation
            # This is required or else the `fully_shard` -> `_move_states_to_device` uses the original memory address
            # for the sharded parameters, and we get a weird/undefined behavior.
            param_group["params"][i] = torch.empty_like(p)

            # We save the original data_ptr, so we can swap back the parameters later
            param_group["params"][i].data_ptr = p.data_ptr()


def swap_back_optimizer_params(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, old_named_parameter_pointers: dict[str, int]
):
    """
    This function is the counterpart of `replace_optimizer_params`. It is called after `fully_shard` being applied to
    the model. It swaps the parameters of the optimizer to their sharded counterparts.
    It is done using the `data_ptr` mapping prepared in `replace_optimizer_params` and `get_named_parameters`.

    Args:
        model (`torch.nn.Module`): Model instance to get the new named parameters from
        optimizer (`torch.optim.Optimizer`): Optimizer instance to swap the parameters of
        old_named_parameter_pointers (`dict[str, int]`): Dictionary mapping the original parameter names: data_ptrs to the new ones
    """
    # We get the new named parameters after `fully_shard` being applied
    # We don't drop the references as we need the sharded parameters now
    new_named_parameters = get_named_parameters(model, drop_refs=False)

    # We create a mapping from the original data_ptr to the new sharded param corresponding to it
    mapping = {p: new_named_parameters[n] for n, p in old_named_parameter_pointers.items()}

    for param_group in optimizer.param_groups:
        # We swap the parameters of the optimizer to the new sharded ones
        param_group["params"] = [mapping[p.data_ptr] for p in param_group["params"]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the benchmarking results.",
    )
    parser.add_argument(
        "--save_memory_snapshot",
        action="store_true",
        default=False,
        help="If True, `torch.cuda.memory._dump_snapshot` will be used to additionaly save the memory trace.",
    )
    ######################
    # Training arguments #
    ######################
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for the training loop.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="The maximum sequence length to use with the model.",
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use.",
    )
    return parser.parse_args()


def prepare_dataloader(tokenizer, args, accelerator: Accelerator) -> DataLoader:
    dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    block_size = min(tokenizer.model_max_length, args.block_size)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)
    dataset = dataset.select(range(int(len(dataset) * args.dataset_fraction)))

    def collate_fn(examples):
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )(examples)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    return dataloader


def get_model(model_name: str):
    # We reguire model to be loaded in fp32, otherwise benchmarks don't match as accelerate does upcasting of parameters to fp32
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
    model = AutoModelForCausalLM.from_config(config)
    return model


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_torch(
    args, config: dict, post_shard_optimizer: bool = False, apply_optimizer_fix: bool = False
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, Accelerator]:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(SEED)
    is_fixed = "fixed" if apply_optimizer_fix else "not_fixed"
    is_post_shard = "optimizer_after_fsdp" if post_shard_optimizer else "optimizer_before_fsdp"
    run_name = f"torch_{is_post_shard}" if post_shard_optimizer else f"torch_{is_post_shard}_{is_fixed}"

    tokenizer = get_tokenizer(config["model_name"])
    train_dataloader = prepare_dataloader(tokenizer, args, accelerator)

    memory_tracker = MemoryTracker(accelerator.device, args.output_dir, run_name, args.save_memory_snapshot)
    memory_tracker.start()

    model = get_model(config["model_name"])
    optimizer = None

    if not post_shard_optimizer:
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

        if apply_optimizer_fix:
            # We drop the references to the original parameters, so that `fully_shard` can trigger a new allocation
            # Then we get the `module_name: data_ptr` mapping, so we can swap back the parameters later
            old_named_parameters = get_named_parameters(model, drop_refs=True)

            # We replace the parameters of the optimizer with empty tensors, so that `fully_shard` can trigger a new allocation
            # We also change the `data_ptr` of the parameters to the original ones, so we can swap back the parameters later
            replace_optimizer_params(optimizer)

    for module in model.modules():
        if isinstance(module, Qwen2DecoderLayer):
            fully_shard(module, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # We do this to imitate how accelerate forces outputs to be in fp32 via `convert_outputs_to_fp32`
    autocast_context = torch.autocast(device_type=accelerator.state.device.type, dtype=torch.bfloat16)
    model_forward_func = model.forward.__func__
    new_forward = autocast_context(model_forward_func)
    model.forward = MethodType(new_forward, model)
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)

    if post_shard_optimizer:
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    if not post_shard_optimizer and apply_optimizer_fix:
        # We swap back the parameters of the optimizer to the original ones
        swap_back_optimizer_params(model, optimizer, old_named_parameters)

    return model, optimizer, train_dataloader, accelerator, memory_tracker


def prepare_accelerate(
    args, config: dict
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, Accelerator]:
    if is_initialized():
        AcceleratorState()._reset_state(True)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen2DecoderLayer"],
    )
    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        mixed_precision="bf16",
    )
    set_seed(SEED)

    tokenizer = get_tokenizer(config["model_name"])
    train_dataloader = prepare_dataloader(tokenizer, args, accelerator)

    memory_tracker = MemoryTracker(accelerator.device, args.output_dir, "accelerate", args.save_memory_snapshot)
    memory_tracker.start()

    model = get_model(config["model_name"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    model, optimizer = accelerator.prepare(model, optimizer)

    return model, optimizer, train_dataloader, accelerator, memory_tracker
