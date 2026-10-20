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

"""Train a tiny causal LM on WikiText and save the loss of each optimizer step to `losses.json`.

Every run trains for `NUM_STEPS` optimizer steps of `GLOBAL_BATCH_SIZE` samples, however the samples of a step are split
across processes and gradient accumulation steps. All samples have `SEQ_LEN` tokens, so they have the same number of
targets and the loss of a step doesn't depend on that split.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator


MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
NUM_STEPS, GLOBAL_BATCH_SIZE, SEQ_LEN = 8, 8, 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--mixed_precision", default="no")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--resume_from_checkpoint")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Tokenize WikiText as one long text, then cut it into samples of `SEQ_LEN` tokens.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    token_ids = tokenizer("\n\n".join(dataset["text"]))["input_ids"]
    num_samples = NUM_STEPS * GLOBAL_BATCH_SIZE
    samples = torch.tensor(token_ids[: num_samples * SEQ_LEN]).view(num_samples, SEQ_LEN)

    # Each process gets an equal share of the samples of a step, split again into accumulation steps.
    batch_size = GLOBAL_BATCH_SIZE // (accelerator.num_processes * args.gradient_accumulation_steps)
    dataloader = DataLoader(samples, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPES[args.dtype])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        dataloader = accelerator.skip_first_batches(dataloader, accelerator.step)

    model.train()
    losses, step_losses = [], []
    for batch in dataloader:
        with accelerator.accumulate(model):
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        step_losses.append(loss.detach())
        if accelerator.sync_gradients:
            losses.append(accelerator.gather(torch.stack(step_losses)).mean().item())
            step_losses = []
            if args.save_steps and len(losses) % args.save_steps == 0:
                accelerator.save_state(os.path.join(args.output_dir, f"step_{len(losses)}"))

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "losses.json"), "w") as f:
            json.dump(losses, f)
    accelerator.end_training()


if __name__ == "__main__":
    main()
