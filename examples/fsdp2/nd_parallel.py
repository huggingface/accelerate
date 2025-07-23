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
Example of training with Context Parallel using FSDP2 via Accelerate.
This example demonstrates how to use Accelerate's context_parallel feature for efficient long sequence training.
"""

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from accelerate.utils.dataclasses import ParallelismConfig
from utils import PerformanceTracker, create_collate_fn, get_dataset, gpu_memory_usage_all, setup_tokenizer


MODEL_ID = "NousResearch/Llama-3.2-1B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--fsdp2-cls-name-to-wrap", type=str, default="LlamaDecoderLayer")
    parser.add_argument("--dp-replicate-size", type=int, default=1)
    parser.add_argument("--dp-shard-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-save-dir", type=str, default="./outputs")
    return parser.parse_args()

def print_rank_zero(str):
    if dist.get_rank() == 0:
        print(str)


def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    set_seed(42)

    if args.model:
        model_id = args.model
    else:
        model_id = MODEL_ID

    model_kwargs = {}
    accelerator_kwargs = {}

    parallelism_config = ParallelismConfig(
        dp_replicate_size=args.dp_replicate_size,
        dp_shard_size=args.dp_shard_size,
        tp_size=args.tp_size,
    )

    if parallelism_config.fsdp_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[args.fsdp2_cls_name_to_wrap],
            reshard_after_forward=True,
            activation_checkpointing=True,
        )
        accelerator_kwargs["fsdp_plugin"] = fsdp2_plugin

    accelerator = Accelerator(
        mixed_precision="bf16",
        parallelism_config=parallelism_config,
        **accelerator_kwargs,
    )

    if args.tp_size > 1:
        model_kwargs["tp_size"] = args.tp_size
        model_kwargs["tp_plan"] = "auto"
        model_kwargs["device_mesh"] = accelerator.torch_device_mesh

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **model_kwargs,
    )
    accelerator.print("Memory usage after model load")
    accelerator.print(gpu_memory_usage_all())
    accelerator.print(model.model.layers[0].self_attn.q_proj.weight)
    accelerator.print("="* 20)
    tokenizer = setup_tokenizer(model_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.print("Memory usage after model prepare")
    accelerator.print(gpu_memory_usage_all())
    accelerator.print(model.model.layers[0].self_attn.q_proj.weight)
    accelerator.print("="* 20)

    dataset = get_dataset(accelerator, tokenizer, args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(100, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=10)

    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        batch_tokens = batch["input_ids"].shape[1]
        metrics = performance_tracker.step(batch_tokens)

        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        log_metrics = {"loss": loss.item()}

        if "warmup_completed" in metrics:
            accelerator.print("Warm up completed! Starting performance tracking...")
        elif metrics:
            print_msg += f" | Average steps/s: {metrics['steps_per_second']:.2f} | Average tokens/s: {metrics['tokens_per_second']:.2f}\n"
            print_msg += (
                f"\tMemory (GB): active={metrics['peak_memory_active']:.1f}, "
                f"alloc={metrics['peak_memory_alloc']:.1f}, "
                f"reserved={metrics['peak_memory_reserved']:.1f}"
            )
        if step % 10 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)

        accelerator.log(log_metrics)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")

    model.save_pretrained(args.model_save_dir)
    accelerator.print(f"Model saved to {args.model_save_dir}")


if __name__ == "__main__":
    main()

"""

MODEL_ID = "NousResearch/Llama-3.2-1B"

###############################################################################################
# baseline FSDP
accelerate launch --num_processes 8 nd_parallel.py --dp-shard-size 8 --dp-replicate-size 1
...
Memory usage after model prepare
{'peak_memory_active': 0.7779741287231445, 'peak_memory_alloc': 0.7779741287231445, 'peak_memory_reserved': 0.90234375}


###############################################################################################
# HSDP
accelerate launch --num_processes 8 nd_parallel.py --dp_shard_size 2 --dp_replicate_size 4
...
Memory usage after model prepare
{'peak_memory_active': 1.6411805152893066, 'peak_memory_alloc': 1.6411805152893066, 'peak_memory_reserved': 1.76953125}

###############################################################################################
# HSDP
accelerate launch --num_processes 8 nd_parallel.py --dp_shard_size 4 --dp_replicate_size 2
...
Memory usage after model prepare
{'peak_memory_active': 1.0664420127868652, 'peak_memory_alloc': 1.0664420127868652, 'peak_memory_reserved': 1.212890625}

###############################################################################################
# FSDP with TP
accelerate launch --num_processes 8 nd_parallel.py --dp-shard-size 4 --dp-replicate-size 1 --tp-size 2


################################################################################################
# Pure TP
accelerate launch --num_processes 8 nd_parallel.py --dp-shard-size 1 --dp-replicate-size 1 --tp-size 8
Memory usage after model load
{'peak_memory_active': 3.93613862991333, 'peak_memory_alloc': 3.93613862991333, 'peak_memory_reserved': 4.009765625}
Memory usage after model prepare
{'peak_memory_active': 3.93613862991333, 'peak_memory_alloc': 3.93613862991333, 'peak_memory_reserved': 4.009765625}

"""
