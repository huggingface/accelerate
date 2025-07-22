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
from accelerate.utils.dataclasses import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from utils import PerformanceTracker, create_collate_fn, get_dataset, setup_tokenizer, gpu_memory_usage_all


MODEL_ID = "NousResearch/Llama-3.2-1B"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-replicate-size", type=int, default=1)
    parser.add_argument("--dp-shard-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    set_seed(42)

    model_kwargs = {}
    accelerator_kwargs = {}

    parallelism_config = ParallelismConfig(
        dp_replicate_size = args.dp_replicate_size,
        dp_shard_size = args.dp_shard_size,
        tp_size = args.tp_size,
    )
    
    if parallelism_config.fsdp_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
            reshard_after_forward=True,
            activation_checkpointing=True,
        )
        accelerator_kwargs["fsdp_plugin"] = fsdp2_plugin

    accelerator = Accelerator(
        mixed_precision="bf16",
        parallelism_config=parallelism_config,
        **accelerator_kwargs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto" if args.tp_size > 1 else None,
        device_mesh=accelerator.torch_device_mesh if args.tp_size > 1 else None,
        **model_kwargs,
    )
    print("Memory usage after model load")
    print(gpu_memory_usage_all())
    tokenizer = setup_tokenizer(MODEL_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)
    print("Memory usage after model prepare")
    print(gpu_memory_usage_all())
    exit()

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


if __name__ == "__main__":
    main()

"""
###############################################################################################
# baseline FSDP
accelerate launch --num_processes 8 nd_parallel.py --dp_shard_size 8 --dp_replicate_size 1
Step 10/100, Loss: 6.5720 | Average steps/s: 3.27 | Average tokens/s: 418.59
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 20/100, Loss: 5.3464 | Average steps/s: 3.27 | Average tokens/s: 418.25
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 30/100, Loss: 5.1396 | Average steps/s: 3.27 | Average tokens/s: 418.78
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 40/100, Loss: 5.2014 | Average steps/s: 3.28 | Average tokens/s: 419.30
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 50/100, Loss: 4.7968 | Average steps/s: 3.28 | Average tokens/s: 419.64
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 60/100, Loss: 4.5652 | Average steps/s: 3.28 | Average tokens/s: 419.87
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 70/100, Loss: 4.8120 | Average steps/s: 3.28 | Average tokens/s: 420.04
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 80/100, Loss: 4.2034 | Average steps/s: 3.28 | Average tokens/s: 420.23
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 90/100, Loss: 4.5770 | Average steps/s: 3.28 | Average tokens/s: 420.34
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2
Step 99/100, Loss: 4.3255 | Average steps/s: 3.28 | Average tokens/s: 420.44
        Memory (GB): active=11.6, alloc=11.6, reserved=12.2

###############################################################################################

accelerate launch --num_processes 8 nd_parallel.py --dp_shard_size 4 --dp_replicate_size 2


###############################################################################################

accelerate launch --num_processes 8 nd_parallel.py --dp_shard_size 2 --dp_replicate_size 4


"""