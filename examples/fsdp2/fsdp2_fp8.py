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
import time

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, FullyShardedDataParallelPlugin, TorchDynamoPlugin, set_seed


WARMUP_STEPS = 10

MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-length", type=int, default=8192, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to train for")
    parser.add_argument("--bf16", action="store_true", help="Train in bf16 precision")
    parser.add_argument("--log-with", type=str, default="wandb", help="Log with wandb or tensorboard")

    return parser.parse_args()


def get_model_flops_per_token(model: AutoModelForCausalLM, args: argparse.Namespace) -> float:
    """
    Get the number of flops per token for the model.

    Args:
        model (AutoModelForCausalLM): Model to get the flops for
    """
    cfg = model.config

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # MLP: 3 matmuls
    mlp_flops = 18 * cfg.hidden_size * cfg.intermediate_size

    # Attn (w/o dotproduct)
    attn_flops = 12 * head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads)

    # attn (dotproduct) - this scales quadratically with sequence length, therefore we have to account for it here too
    attn_dotproduct_flops = 12 * cfg.num_attention_heads * head_dim * args.sequence_length

    # we also ignore embeddings and layernorms, etc
    return (mlp_flops + attn_flops + attn_dotproduct_flops) * cfg.num_hidden_layers


def get_dataset(accelerator: Accelerator, tokenizer: AutoTokenizer, seq_len: int) -> Dataset:
    """
    Load and prepare TinyStories dataset.

    Args:
        accelerator (Accelerate): Accelerate accelerator instance
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        seq_len (int): Sequence length for the dataset

    Returns:
        Dataset: Packed dataset
    """

    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:50%]")

    def tokenize_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=seq_len,
            return_tensors=None,
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def create_packed_sequences(examples):
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        num_sequences = len(all_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            full_sequence = all_tokens[start_idx:end_idx]
            packed_input_ids.append(full_sequence[:-1])
            packed_labels.append(full_sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    with accelerator.main_process_first():
        packed_dataset = tokenized_dataset.map(
            create_packed_sequences,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            batch_size=1000,
        )

    return packed_dataset.shuffle(seed=42)


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
    fsdp2_plugin.set_mixed_precision("bf16")

    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        use_regional_compilation=True,  # We use regional compilation to compile the model way faster
    )

    fp8_config = Float8LinearConfig(
        enable_fsdp_float8_all_gather=True,  # extra saving by gathering parameters in fp8 and upcasting after
        force_recompute_fp8_weight_in_bwd=True,
    )

    kwargs = []
    if not args.bf16:
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
        config=AutoConfig.from_pretrained(MODEL_ID, use_cache=False),
        torch_dtype=torch.bfloat16,
        # use_cache=False,  # Important to set `use_cache=False`
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)

    dataset = get_dataset(accelerator, tokenizer, args.sequence_length)

    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    # We keep batch size at 1, as it is basically the same as sequence length, which we use instead
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(args.num_steps, len(dataloader))
    num_tokens = 0
    is_in_warmup = True
    model_flops_per_token = get_model_flops_per_token(model, args)

    accelerator.print(f"Warming up for {WARMUP_STEPS} steps...")

    for step, batch in enumerate(dataloader):
        if step == WARMUP_STEPS:
            accelerator.print("Warm up completed! Starting training")
            start_time = time.perf_counter()
            num_tokens = 0
            is_in_warmup = False

        if step >= total_num_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        steps_from_warmup = step - WARMUP_STEPS
        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        metrics = {"loss": loss.item()}

        if not is_in_warmup and steps_from_warmup > 0:
            num_tokens += batch["input_ids"].shape[1]
            total_time = time.perf_counter() - start_time
            tps = num_tokens / total_time
            tflops = num_tokens * model_flops_per_token / (total_time * 1e12)

            # it's rather hard to get a good estimate of MFU as we train with FP8, so both FP8 and BF16 tensor cores are used, therefore we just report TFLOPS (Tera floating point operations per second)
            # Given H100 SXM, the theoretical peak flops are ~990 TFLOPS for bf16 and ~1980 TFLOPS for fp8 [https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306]
            # This is WITH sparsity, so we divide by 2 to get the answer w/o sparsity
            print_msg += f", Average steps/s: {steps_from_warmup / total_time:.2f}, TPS per device: {tps:.2f}, TFLOPS per device: {tflops:.2f}"
            metrics.update(
                {
                    "steps_per_second": steps_from_warmup / total_time,
                    "tps_per_device": tps,
                    "tflops_per_device": tflops,
                }
            )

        if steps_from_warmup % 10 == 0 or step == total_num_steps:
            accelerator.print(print_msg)

        accelerator.log(metrics)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
