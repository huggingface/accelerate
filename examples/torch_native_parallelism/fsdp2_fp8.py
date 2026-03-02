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
Minimal example of training with FP8 precision using FSDP2 via Accelerate.
This example demonstrates how to use torchao's Float8LinearConfig with Accelerate's AORecipeKwargs.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, FullyShardedDataParallelPlugin, TorchDynamoPlugin, set_seed
from utils import PerformanceTracker, create_collate_fn, get_dataset, get_model_flops_per_token


WARMUP_STEPS = 10

MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-length", type=int, default=8192, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to train for")
    parser.add_argument("--precision", type=str, default="fp8", choices=["fp8", "bf16"], help="Precision to train in")
    parser.add_argument("--log-with", type=str, default="wandb", help="Log with wandb or tensorboard")

    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    set_seed(42)

    args = parse_args()

    fsdp2_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        cpu_ram_efficient_loading=False,  # CPU RAM efficient loading CANNOT work with fp8 torchao
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
    )
    fsdp2_plugin.set_mixed_precision(args.precision)

    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        use_regional_compilation=True,  # We use regional compilation to compile the model way faster
    )

    fp8_config = Float8LinearConfig(
        enable_fsdp_float8_all_gather=True,  # extra saving by gathering parameters in fp8 and upcasting after
    )

    kwargs = []
    if args.precision == "fp8":
        kwargs = [AORecipeKwargs(config=fp8_config)]

    accelerator = Accelerator(
        fsdp_plugin=fsdp2_plugin,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs,
        log_with=args.log_with,
    )
    accelerator.init_trackers(
        project_name="FSDP2_torchao_fp8",
        config={"sequence_length": args.sequence_length, "num_steps": args.num_steps},
    )

    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(MODEL_ID, use_cache=False),
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dataset = get_dataset(tokenizer, args.sequence_length, accelerator)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    accelerator.wait_for_everyone()

    model.train()

    total_num_steps = min(args.num_steps, len(dataloader))
    model_flops_per_token = get_model_flops_per_token(model, args.sequence_length)
    performance_tracker = PerformanceTracker(warmup_steps=5)

    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        metrics = performance_tracker.step(batch["input_ids"].shape[1], model_flops_per_token)

        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        if "warmup_completed" in metrics:
            accelerator.print("Warm up completed! Starting training")
        elif metrics:
            print_msg += performance_tracker.get_print_message(metrics)

        if step % 10 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)

        accelerator.log(metrics)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
