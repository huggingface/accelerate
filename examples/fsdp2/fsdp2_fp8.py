"""
Minimal example of training with FP8 precision using FSDP2 via Accelerate.
This example demonstrates how to use torchao's Float8LinearConfig with Accelerate's AORecipeKwargs.
"""

import argparse
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, FullyShardedDataParallelPlugin, TorchDynamoPlugin, set_seed


WARMUP_STEPS = 10

# Set high precision for matmul operations
torch.set_float32_matmul_precision("high")

MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-length", type=int, default=8192, help="Sequence length for the dataset")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to train for")
    parser.add_argument("--log-with", type=str, default="wandb", help="Log with wandb or tensorboard")

    return parser.parse_args()


def get_dataset(accelerator, tokenizer, seq_len):
    """Load and prepare TinyStories dataset with packing."""
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")

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


def warmup(model, dataloader, accelerator):
    accelerator.print("Warming up...")

    num_steps = 10
    for i, batch in enumerate(dataloader):
        model(**batch)
        if i >= num_steps:
            break

    accelerator.print("Warm up completed!")


def main():
    set_seed(42)

    args = parse_args()

    fsdp2_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        cpu_ram_efficient_loading=False,
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
    kwargs = [AORecipeKwargs(config=fp8_config)]

    accelerator = Accelerator(
        fsdp_plugin=fsdp2_plugin,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs,
        log_with=[args.log_with],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Important to set `use_cache=False`
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

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    dataloader = accelerator.prepare(dataloader)

    model.train()
    total_num_steps = max(args.num_steps, len(dataloader))

    accelerator.wait_for_everyone()

    accelerator.print(f"Warming up for {WARMUP_STEPS} steps...")
    num_tokens = 0

    is_in_warmup = True

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

        print_msg = f"Step {step + 1}/{total_num_steps}, Loss: {loss.item():.4f}"

        if not is_in_warmup:
            steps_from_warmup = step - WARMUP_STEPS
            num_tokens += batch["input_ids"].shape[1]
            total_time = time.perf_counter() - start_time
            print_msg += f", Average steps/s: {steps_from_warmup / total_time:.2f}, TPS per device: {num_tokens / total_time:.2f}"

        if step % 10 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)
        accelerator.log({"loss": loss.item()})

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
