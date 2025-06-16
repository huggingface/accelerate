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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from utils import PerformanceTracker, create_collate_fn, get_dataset, setup_tokenizer


MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=128_000, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--log-with", type=str, default="wandb", help="Logging service to use")
    parser.add_argument("--cp-size", type=int, default=8, help="Context parallel size")
    parser.add_argument("--cp-comm-strategy", type=str, default="allgather", help="Context parallel shard rotation")
    return parser.parse_args()


def training_step(batch, model, optimizer, accelerator: Accelerator):
    """
    Perform a single training step with context parallel.

    Args:
        batch: Input batch containing input_ids and labels
        model: The model to train
        optimizer: Optimizer
        accelerator: Accelerator instance

    Returns:
        loss: Training loss
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Use context parallel for efficient long sequence processing
    with accelerator.context_parallel(
        buffers=[input_ids, labels],
        buffer_seq_dims=[1, 1],  # Sequence dimension is dimension 1 for both tensors
        no_restore_buffers={input_ids, labels},  # Don't restore these buffers after forward pass
    ):
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        return loss


def main():
    set_seed(42)
    args = parse_args()

    # Configure FSDP2 plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
        cpu_ram_efficient_loading=True,
        activation_checkpointing=True,
        fsdp_version=2,
        cp_size=args.cp_size,
        cp_comm_strategy=args.cp_comm_strategy,
    )

    # Initialize accelerator
    accelerator = Accelerator(
        log_with=args.log_with,
        fsdp_plugin=fsdp_plugin,
        mixed_precision="bf16",
    )

    accelerator.init_trackers(
        project_name="FSDP2_context_parallel",
        config={
            "sequence_length": args.sequence_length,
            "num_steps": args.num_steps,
            "cp_size": args.cp_size,
            "cp_comm_strategy": args.cp_comm_strategy,
        },
    )

    # Prepare model and optimizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = setup_tokenizer(MODEL_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)

    accelerator.print("Preparing dataset... this might take a while")
    dataset = get_dataset(accelerator, tokenizer, args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(args.num_steps, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=10)

    accelerator.print(f"Starting training with context parallel for {total_num_steps} steps...")
    accelerator.print(f"Sequence length: {args.sequence_length}")
    accelerator.print("Warming up for 10 steps...")

    accelerator.print(
        "Each step takes ~10 seconds with default settings on 8x H100 SXM GPUs, seeing logs takes a while"
    )
    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        # get number of tokens before context_parallel shards the batch
        batch_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

        loss = training_step(batch, model, optimizer, accelerator)

        # each batch gets the same data, we divide by the number of processes to get the number of tokens per process
        metrics = performance_tracker.step(batch_tokens // accelerator.num_processes)

        log_metrics = {"loss": loss.item()}

        if "warmup_completed" in metrics:
            accelerator.print("Warmup completed! Starting performance tracking...")
        elif metrics:
            log_metrics.update(
                {
                    "tokens_per_second": int(metrics["tokens_per_second"]),
                    "steps_per_second": metrics["steps_per_second"],
                }
            )

        if (step % 10 == 0 or step == total_num_steps - 1) and metrics:
            accelerator.print(
                f"Step {step}/{total_num_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Tokens/s: {int(metrics['tokens_per_second'])} | "
                f"Steps/s: {metrics['steps_per_second']:.2f} | "
            )

        accelerator.log(log_metrics)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
