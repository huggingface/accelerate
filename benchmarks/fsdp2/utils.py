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

import argparse  # noqa: I001

from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
import torch
from accelerate import FullyShardedDataParallelPlugin
from datasets import load_dataset
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from accelerate import Accelerator

from measure_utils import MemoryTracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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
        default=8,
        help="Batch size for the training loop.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="The maximum sequence length to use with the model.",
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


def replace_optimizer_params(optimizer):
    for param_group in optimizer.param_groups:
        for i, p in enumerate(param_group["params"]):
            param_group["params"][i] = torch.empty_like(p)
            param_group["params"][i].data_ptr = p.data_ptr()


def swap_back_optimizer_params(
    accelerator: Accelerator, model: torch.nn.Module, optimizer: torch.optim.Optimizer, old_named_parameters: dict
):
    new_named_parameters = accelerator._get_named_parameters(model)

    mapping = {p: new_named_parameters[n] for n, p in old_named_parameters.items()}

    for param_group in optimizer.param_groups:
        param_group["params"] = [mapping[p.data_ptr] for p in param_group["params"]]


def get_model_and_tokenizer(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_torch(
    args, config: dict, post_shard_optimizer: bool = False, apply_optimizer_fix: bool = False
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, Accelerator]:
    from torch.distributed.fsdp import MixedPrecisionPolicy

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42)
    is_fixed = "fixed" if apply_optimizer_fix else "not_fixed"
    is_post_shard = "post_shard" if post_shard_optimizer else "pre_shard"
    run_name = f"torch_{is_post_shard}" if post_shard_optimizer else f"torch_{is_post_shard}_{is_fixed}"
    memory_tracker = MemoryTracker(accelerator, args.output_dir, run_name, args.save_memory_snapshot)
    memory_tracker.start()
    model, tokenizer = get_model_and_tokenizer(config["model_name"])
    train_dataloader = prepare_dataloader(tokenizer, args, accelerator)
    optimizer = None

    if not post_shard_optimizer:
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

        if apply_optimizer_fix:
            old_named_parameters = accelerator._get_named_parameters(model, drop_refs=True)
            replace_optimizer_params(optimizer)

    for module in model.modules():
        if isinstance(module, Qwen2DecoderLayer):
            fully_shard(module, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    if post_shard_optimizer:
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    if not post_shard_optimizer and apply_optimizer_fix:
        swap_back_optimizer_params(accelerator, model, optimizer, old_named_parameters)

    return model, optimizer, train_dataloader, accelerator, memory_tracker


def prepare_accelerate(
    args, config: dict
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, Accelerator]:
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
    set_seed(42)
    memory_tracker = MemoryTracker(accelerator, args.output_dir, "accelerate", args.save_memory_snapshot)
    memory_tracker.start()
    model, tokenizer = get_model_and_tokenizer(config["model_name"])
    train_dataloader = prepare_dataloader(tokenizer, args, accelerator)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    model, optimizer = accelerator.prepare(model, optimizer)

    return model, optimizer, train_dataloader, accelerator, memory_tracker
