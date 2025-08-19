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
Example of training with ND parallel using accelerate's ParallelismConfig
"""

import argparse
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from utils import (
    PerformanceTracker,
    create_collate_fn,
    get_dataset,
    setup_tokenizer,
)

MODEL_ID = "Qwen/Qwen2.5-7B"
# MODEL_ID = "axolotl-ai-co/gpt-oss-120b-dequantized"

def forward(model, batch, optimizer, accelerator: Accelerator):
    # We need both labels and shift_labels, as the loss computation in the model is hidden behind `if labels is not None`, but the loss computation
    # itself prioritzes shift_labels (if provided) which are the correct ones (due to labels being wrong if cp enabled)
    buffers = [batch["input_ids"], batch["shift_labels"], batch["labels"]]
    with accelerator.maybe_context_parallel(
        buffers=buffers, buffer_seq_dims=[1, 1, 1], no_restore_buffers=set(buffers)
    ):
        # To get the proper loss value, we need to average across devices that are participating in data parallel/context parallel training
        # As for DP we have a different batch on each device and for CP we essentially have a different part of sequences on each device
        # I.e. with causal modelling and seq_len 1024, this dimension becomes another batch dimension of sorts
        loss_reduce_grp = (
            accelerator.torch_device_mesh["dp_cp"].get_group()
            if accelerator.parallelism_config.dp_cp_dim_names
            else None
        )
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=loss_reduce_grp)

    return loss


def train():
    parallelism_config = ParallelismConfig(
        dp_shard_size=2,
    )

    fsdp2_plugin = None
    if parallelism_config.dp_shard_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            # transformer_cls_names_to_wrap=["GptOssDecoderLayer"],
            transformer_cls_names_to_wrap=["Qwen2DecoderLayer"],
            state_dict_type="SHARDED_STATE_DICT",
            cpu_offload=True
            
        )

    accelerator = Accelerator(
        mixed_precision="bf16", parallelism_config=parallelism_config, fsdp_plugin=fsdp2_plugin
    )
    if accelerator.is_main_process:
        device_map = "cpu"
    else:
        device_map="meta"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)
    print(torch.distributed.get_rank(), gpu_memory_usage_all())

def gpu_memory_usage_all(device=0):
    device = torch.device(f"cuda:{device}")
    _BYTES_IN_GIB = 1024**3
    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / _BYTES_IN_GIB
    peak_memory_alloc = torch.cuda.max_memory_allocated(device) / _BYTES_IN_GIB
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / _BYTES_IN_GIB
    memory_stats = {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_memory_alloc,
        "peak_memory_reserved": peak_memory_reserved,
    }
    torch.cuda.reset_peak_memory_stats(device)

    return memory_stats


if __name__ == "__main__":
    set_seed(42)
    train()
