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
import contextlib

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, TorchTensorParallelPlugin, set_seed
from utils import PerformanceTracker, create_collate_fn, get_dataset, setup_tokenizer


MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-tp", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--apply-fsdp", action="store_true")
    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    set_seed(42)

    plugin_kwargs = {}
    model_kwargs = {}
    accelerator_kwargs = {}

    if args.apply_tp and args.apply_fsdp:
        device_mesh = init_device_mesh(mesh_dim_names=("dp_shard", "tp"), mesh_shape=(4, 2), device_type="cuda")
        plugin_kwargs["device_mesh"] = device_mesh
        model_kwargs["device_mesh"] = device_mesh

    if args.apply_tp:
        model_kwargs["tp_plan"] = "auto"
        model_kwargs["tp_size"] = 2

    if args.apply_fsdp:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
            **plugin_kwargs,
        )
        fsdp2_plugin.set_mixed_precision("bf16")
        accelerator_kwargs["fsdp_plugin"] = fsdp2_plugin

    if not args.apply_fsdp and args.apply_tp:
        tp_plugin = TorchTensorParallelPlugin(tp_size=2)
        accelerator_kwargs["torch_tp_plugin"] = tp_plugin

    accelerator = Accelerator(
        log_with=["wandb"],
        **accelerator_kwargs,
    )
    accelerator.init_trackers(
        project_name="fsdp2-tp",
        config={"apply_tp": args.apply_tp, "apply_fsdp": args.apply_fsdp},
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **model_kwargs,
    )

    tokenizer = setup_tokenizer(MODEL_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)

    dataset = get_dataset(accelerator, tokenizer, 256)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(1000, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=10)

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof = contextlib.nullcontext()

    with prof:
        for step, batch in enumerate(dataloader):
            if step >= total_num_steps:
                break

            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if args.profile:
                break

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

            batch_tokens = batch["input_ids"].shape[1]
            metrics = performance_tracker.step(batch_tokens)

            print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
            log_metrics = {"loss": loss.item()}

            if "warmup_completed" in metrics:
                accelerator.print("Warm up completed! Starting performance tracking...")
            elif metrics:
                print_msg += f" | Average steps/s: {metrics['steps_per_second']:.2f}"

            if step % 10 == 0 or step == total_num_steps - 1:
                accelerator.print(print_msg)

            accelerator.log(log_metrics)

    if args.profile and accelerator.is_main_process:
        trace_name = f"trace_{'tp' if args.apply_tp else 'no_tp'}.json"
        prof.export_chrome_trace(trace_name)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
