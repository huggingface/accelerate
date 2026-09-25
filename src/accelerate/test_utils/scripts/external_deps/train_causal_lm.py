# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Train a tiny causal LM for the DDP training comparisons."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import set_seed


def train_reference(model, optimizer, dataloader, mixed_precision, device):
    """Single-device PyTorch; bypass Accelerate's preparation, backward and optimizer wrappers."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
    losses = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = model(input_ids=batch, labels=batch).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def train_with_accelerate(model, optimizer, dataloader, accelerator):
    losses, window_losses = [], []

    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            if accelerator.sync_gradients:
                # Equal shifted-target counts make this a global effective-batch mean.
                window_loss = torch.stack(window_losses).mean()
                losses.append(accelerator.reduce(window_loss, reduction="mean").item())
                window_losses.clear()

    return losses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = None
    if not args.reference:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    device = torch.device("cuda:0") if args.reference else accelerator.device

    set_seed(1337)
    # Keep FP32 matrix multiplies in full precision for the loss comparison.
    torch.set_float32_matmul_precision("highest")

    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Fix the attention implementation; training does not need a generation cache.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
        use_cache=False,
    )
    model.train()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision)
    if dtype:
        # Returned logits are upcast by Accelerate; observe an operation inside forward.
        def check_compute_dtype(module, inputs, output):
            assert output.dtype == dtype, f"Expected {dtype} compute, got {output.dtype}"

        linear = next(module for module in model.modules() if isinstance(module, torch.nn.Linear))
        linear.register_forward_hook(check_compute_dtype)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_attention_mask=False)["input_ids"]

    block_size, num_blocks = 32, 80
    # Full blocks give equal shifted-target counts, so averaging microbatch losses
    # matches the full-batch token mean. All ten global batches are complete.
    input_ids = torch.tensor(tokens[: num_blocks * block_size]).reshape(num_blocks, block_size)
    dataloader = DataLoader(input_ids, batch_size=args.batch_size, shuffle=False)

    if args.reference:
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if args.reference:
        losses = train_reference(model, optimizer, dataloader, args.mixed_precision, device)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        losses = train_with_accelerate(model, optimizer, dataloader, accelerator)

    # Revisit the first global batch to observe learning, including the final update.
    first_global_batch = input_ids[:8].to(device)
    # Accelerate's prepared model handles autocast inside forward; only the reference needs it here.
    context = torch.autocast("cuda", dtype=dtype) if args.reference and dtype is not None else nullcontext()
    with torch.no_grad(), context:
        final_loss = model(input_ids=first_global_batch, labels=first_global_batch).loss.item()

    if args.reference or accelerator.is_main_process:
        results = {
            "losses": losses,
            "final_loss": final_loss,
            "world_size": 1 if args.reference else accelerator.num_processes,
        }
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if accelerator:
        accelerator.end_training()


if __name__ == "__main__":
    main()
