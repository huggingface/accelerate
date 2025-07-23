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

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from accelerate.utils.dataclasses import ParallelismConfig
from accelerate.utils.fsdp_utils import save_fsdp_optimizer
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
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="Whether to save the model after training."
    )
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        default=False,
        help="Whether to save the optimizer state after training.",
    )
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

    device_mesh = parallelism_config.build_device_mesh("cuda")

    if args.tp_size > 1:
        model_kwargs["tp_size"] = args.tp_size
        model_kwargs["tp_plan"] = "auto"
        model_kwargs["device_mesh"] = device_mesh

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **model_kwargs,
    )

    partial_state = PartialState()
    partial_state.device_mesh = device_mesh
    partial_state.parallelism_config = parallelism_config

    if parallelism_config.fsdp_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[args.fsdp2_cls_name_to_wrap],
            reshard_after_forward=True,
            activation_checkpointing=True,
            state_dict_type="FULL_STATE_DICT",
        )
        accelerator_kwargs["fsdp_plugin"] = fsdp2_plugin

    accelerator = Accelerator(
        mixed_precision="no",
        **accelerator_kwargs,
    )

    accelerator.print("Memory usage after model load")
    accelerator.print(gpu_memory_usage_all())
    accelerator.print(model.model.layers[0].self_attn.q_proj.weight)
    accelerator.print("=" * 20)
    tokenizer = setup_tokenizer(model_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.print("Memory usage after model prepare")
    accelerator.print(gpu_memory_usage_all())
    accelerator.print(model.model.layers[0].self_attn.q_proj.weight)
    accelerator.print("=" * 20)

    dataset = get_dataset(accelerator, tokenizer, args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(100, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=10)

    accelerator.print("Starting training...")
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
    if parallelism_config.fsdp_enabled and args.save_optimizer:
        accelerator.print("Saving optimizer state...")
        save_fsdp_optimizer(
            fsdp2_plugin,
            accelerator,
            optimizer,
            model,
            args.model_save_dir + "/opt",
        )
        accelerator.print("Optimizer state saved.")
    accelerator.print("Saving model state...")
    if args.save_model:
        model.save_pretrained(args.model_save_dir)
        accelerator.print(f"Model saved to {args.model_save_dir}")


if __name__ == "__main__":
    main()
