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


MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-replicate-size", type=int, default=1)
    parser.add_argument("--dp-shard-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint-frequency", type=int, default=100)
    parser.add_argument("--model-name", type=str, default=MODEL_ID)

    return parser.parse_args()


def forward(model, batch, optimizer, accelerator: Accelerator):
    batch["position_ids"] = torch.arange(0, batch["input_ids"].size(1), device=batch["input_ids"].device).unsqueeze(0)
    # We need both labels and shift_labels, as the loss computation in the model is hidden behind `if labels is not None`, but the loss computation
    # itself prioritzes shift_labels (if provided) which are the correct ones (due to labels being wrong if cp enabled)
    buffers = [batch["input_ids"], batch["shift_labels"], batch["labels"], batch["position_ids"]]
    with accelerator.maybe_context_parallel(
        buffers=buffers, buffer_seq_dims=[1, 1, 1, 1], no_restore_buffers=set(buffers)
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


def train(args):
    parallelism_config = ParallelismConfig(
        dp_replicate_size=args.dp_replicate_size,
        dp_shard_size=args.dp_shard_size,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
    )

    # FSDP needs extra configuration, so we properly shard the model
    fsdp2_plugin = None
    if parallelism_config.dp_shard_enabled or parallelism_config.cp_enabled:
        fsdp2_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
            state_dict_type="SHARDED_STATE_DICT",
        )

    accelerator = Accelerator(
        log_with=["wandb"], mixed_precision="bf16", parallelism_config=parallelism_config, fsdp_plugin=fsdp2_plugin
    )
    accelerator.init_trackers("nd_parallel_training")

    # If TP was enabled, we need to tell transformers to prepare the model for us
    model_kwargs = (
        {"tp_size": args.tp_size, "tp_plan": "auto", "device_mesh": accelerator.torch_device_mesh}
        if args.tp_size > 1
        else {}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **model_kwargs,
    )
    tokenizer = setup_tokenizer(args.model_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    dataset = get_dataset(tokenizer, args.sequence_length, accelerator)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    total_num_steps = min(args.num_steps, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=5)

    accelerator.print("Starting training...")
    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        loss = forward(model, batch, optimizer, accelerator)

        # We report TPS per device, so we divide by the number of devices in the non-data parallel dimension
        metrics = performance_tracker.step(batch["input_ids"].shape[1] / parallelism_config.non_data_parallel_size)

        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        if "warmup_completed" in metrics:
            accelerator.print("Warm up completed! Starting performance tracking...")
        elif metrics:
            print_msg += performance_tracker.get_print_message(metrics, with_memory=True)

        if step % 10 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)

        if step % args.checkpoint_frequency == 0 and step > 0 and parallelism_config.dp_shard_enabled:
            accelerator.print(f"Saving checkpoint at step {step}...")
            accelerator.save_state(args.save_dir + f"/checkpoint-{step}")

        accelerator.log({"loss": loss.item()})

    accelerator.print("Training completed!")

    model.save_pretrained(args.save_dir + f"/{args.model_name}")
    accelerator.print(f"Model saved to {args.save_dir}/{args.model_name}")
    accelerator.end_training()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    if args.dp_shard_size == 1 and args.tp_size > 1:
        # We currently don't support saving with `save_state` when using only
        # tensor parallelism, fsdp must be enabled
        warnings.warn(
            "Accelerator.save_state() is not yet supported with pure tensor parallel training. Training will work, but intermediate checkpoints will not be saved."
        )
    train(args)
