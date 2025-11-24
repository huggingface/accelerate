
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

# Example of fine-tuning a model on a TPU using FSDPv2, TRL and PEFT.
#
# Run the script with:
# python finetune_lm_tpu.py [--model_id MODEL_ID] [--dataset_id DATASET_ID]
#
# This script has been tested on a TPU v5 litepod-8.

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import argparse

import torch_xla.runtime as xr

# FSDPv2 requires SPMD to be enabled.
xr.use_spmd()


def format_dolly(example, tokenizer):
    """Format Dolly dataset examples using the tokenizer's chat template."""
    user_content = example["instruction"]
    if len(example["context"]) > 0:
        user_content += f"\n\nContext: {example['context']}"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["response"]},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False)


def train(model_id, dataset):
    # Load model with low_cpu_mem_usage to avoid loading full model into CPU memory
    # FSDPv2 will handle sharding across TPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,  # Let FSDP handle device placement
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        if model.config.model_type == "llama":
            # Vanilla Llama models have a finetune gith pad id token
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(f"Cannot get or guess pad token for model {model_id}.")

    if tokenizer.chat_template is None:
        # Set chat template for Llama 3.1 format
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )

    # Try to guess the DecoderLayer class name, based on common model architectures
    transformer_layer_cls_to_wrap = model.model.layers[0].__class__.__name__

    # Get FSDP training arguments
    fsdp_training_args = {
        "fsdp": "full_shard",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [transformer_layer_cls_to_wrap],
            "xla": True,
            "xla_fsdp_v2": True,
            "xla_fsdp_grad_ckpt": True,
        },
    }

    # Set up PEFT LoRA for fine-tuning.
    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
            gradient_checkpointing=False, # Required on TPU, not supported
            max_length=1024,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            max_steps=-1,
            output_dir="./output",
            optim="adafactor",
            logging_steps=1,
            dataloader_drop_last = True,  # Required for FSDPv2.
            dataset_text_field="text",
            packing=True,
            **fsdp_training_args,
    )

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=lambda example: format_dolly(example, tokenizer),
    )

    trainer.train()


# =============================================================================
# Main Function
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")

    parser.add_argument("--model_id", "-m", type=str, default="meta-llama/Llama-3.2-1B", help="Model id to use for training.")
    parser.add_argument("--dataset_id", "-d", type=str, default="databricks/databricks-dolly-15k", help="Dataset id to use for training.")

    args = parser.parse_args()

    # NOTE: this section can be adapted to load any dataset you want.
    dataset_id = args.dataset_id
    dolly_dataset = load_dataset(dataset_id, split="train")

    train(
        model_id=args.model_id,
        dataset=dolly_dataset,
    )
