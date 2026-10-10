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


def register_mixed_precision_check(model, expected_dtype):
    """Check that a Linear layer produces output in the requested mixed precision.

    Training can succeed with similar losses even when mixed precision is
    inactive. Check the output dtype to ensure it was actually used.
    """

    def check_output_dtype(module, inputs, output):
        assert output.dtype == expected_dtype, f"Expected {expected_dtype} output, got {output.dtype}"

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(check_output_dtype)
            return

    raise ValueError("Expected a Linear layer to check mixed-precision output.")


def train_reference(model, optimizer, dataloader, mixed_precision_dtype, device):
    """Single-device PyTorch; bypass Accelerate's preparation, backward and optimizer wrappers."""
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_dtype == torch.float16)
    losses = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=mixed_precision_dtype, enabled=mixed_precision_dtype is not None):
            loss = model(input_ids=batch, labels=batch).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def train_with_accelerate(model, optimizer, dataloader, accelerator):
    """Train with Accelerate and record one global mean loss per optimizer update."""
    losses, window_losses = [], []

    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            if accelerator.sync_gradients:
                # Each microbatch has the same number of next-token targets.
                # Average within this accumulation window, then across processes.
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
    # Explicitly use full FP32 matmul precision for this comparison.
    torch.set_float32_matmul_precision("highest")

    checkpoint = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Training does not reuse the attention cache used for generation.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        use_cache=False,
    )
    model.train()

    mixed_precision_dtype = {
        "no": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]
    if mixed_precision_dtype is not None:
        register_mixed_precision_check(model, mixed_precision_dtype)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, return_attention_mask=False)["input_ids"]

    block_size, num_blocks = 32, 80
    # Each block has 31 next-token targets, so equally sized microbatch losses
    # can be averaged without reweighting. Eight blocks per update give ten complete updates.
    input_ids = torch.tensor(token_ids[: num_blocks * block_size]).reshape(num_blocks, block_size)
    dataloader = DataLoader(input_ids, batch_size=args.batch_size, shuffle=False)

    if args.reference:
        model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    if args.reference:
        losses = train_reference(model, optimizer, dataloader, mixed_precision_dtype, device)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        losses = train_with_accelerate(model, optimizer, dataloader, accelerator)

    # In DDP, both processes revisit the same first eight examples after the final update.
    # This measures progress on training data, not performance on unseen text.
    first_global_batch = input_ids[:8].to(device)
    # Accelerate's prepared model handles autocast inside forward; only the reference needs it here.
    context = (
        torch.autocast("cuda", dtype=mixed_precision_dtype)
        if args.reference and mixed_precision_dtype is not None
        else nullcontext()
    )
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
