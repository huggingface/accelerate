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
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torchao.optim import AdamW4bit, AdamW8bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from utils import PerformanceTracker, create_collate_fn, get_dataset, get_model_flops_per_token


WARMUP_STEPS = 10

MODEL_ID = "meta-llama/Llama-3.2-3B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to train for")
    parser.add_argument("--log-with", type=str, default="wandb", help="Log with wandb or tensorboard")
    parser.add_argument("--run-name", type=str, default=None, help="The name of the run for logging")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA adapters", default=False)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit", default=False)
    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    set_seed(42)

    args = parse_args()

    fsdp2_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
        mixed_precision_policy="bf16",
    )

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,  # required for FSDP(2)
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_cache=False, quantization_config=bnb_config)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )  # use activation checkpointing from FSDP2 instead

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=8,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model, lora_config, autocast_adapter_dtype=False
        )  # keep the adapters in bf16, if you want to remove this, model needs to be loaded in fp32 too

        for n, p in model.named_parameters():
            if any(x in n for x in target_modules):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        model.enable_input_require_grads()

    accelerator = Accelerator(
        fsdp_plugin=fsdp2_plugin,
        log_with=args.log_with,
        mixed_precision="bf16",
    )
    accelerator.init_trackers(
        project_name="FSDP2-PEFT",
        config={
            "sequence_length": args.sequence_length,
            "num_steps": args.num_steps,
            "use-lora": args.use_lora,
            "use_8bit_optim": args.use_8bit_optim,
            "load_in_4bit": args.load_in_4bit,
        },
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
    performance_tracker = PerformanceTracker(warmup_steps=5)

    flops_per_token = get_model_flops_per_token(model, args.sequence_length)

    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        metrics = performance_tracker.step(batch["input_ids"].shape[1], flops_per_token)

        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        if "warmup_completed" in metrics:
            accelerator.print("Warm up completed! Starting training")
        elif metrics:
            print_msg += performance_tracker.get_print_message(metrics)

        if step % 1 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)

        accelerator.log({"loss": loss.item(), **metrics})

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")

    accelerator.save_state()


if __name__ == "__main__":
    main()
