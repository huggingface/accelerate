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

"""Train a causal LM in FP32, optionally saving or resuming after five updates."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.test_utils.causal_lm import load_model_and_data
from accelerate.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--mixed-precision", choices=("no",), default="no")

    checkpoint_args = parser.add_mutually_exclusive_group()
    checkpoint_args.add_argument("--save-checkpoint", type=Path)
    checkpoint_args.add_argument("--resume-from-checkpoint", type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps != 8:
        raise ValueError("This fixture requires eight blocks per complete update.")

    set_seed(1337)
    # Use full FP32 matrix multiplication precision for the loss comparison.
    torch.set_float32_matmul_precision("highest")

    model, input_ids = load_model_and_data()
    dataset = TensorDataset(input_ids, torch.arange(len(input_ids)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Momentum makes loading model weights alone insufficient for correct continuation.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # With two ranks, Accelerate takes two scheduler steps per update; step_size=4 decays every two updates.
    # Restoring only the optimizer's learning rate cannot preserve this schedule.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    checkpoint_update = 5
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        # Each saved update consumed this many local microbatches.
        num_batches_to_skip = checkpoint_update * args.gradient_accumulation_steps
        dataloader = accelerator.skip_first_batches(dataloader, num_batches_to_skip)

    losses, learning_rates, sample_ids = [], [], []
    window_losses, window_sample_ids = [], []

    for batch, batch_sample_ids in dataloader:
        with accelerator.accumulate(model):
            learning_rate = optimizer.param_groups[0]["lr"]
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            window_losses.append(loss.detach())
            window_sample_ids.append(batch_sample_ids)
            if accelerator.sync_gradients:
                window_loss = torch.stack(window_losses).mean()
                losses.append(accelerator.reduce(window_loss, reduction="mean").item())
                learning_rates.append(learning_rate)
                sample_ids.append(sorted(accelerator.gather(torch.cat(window_sample_ids)).cpu().tolist()))
                window_losses.clear()
                window_sample_ids.clear()

                if args.save_checkpoint and len(losses) == checkpoint_update:
                    accelerator.save_state(args.save_checkpoint)
                    break

    # Both processes revisit the first global batch, as in the ordinary training comparison.
    first_global_batch = input_ids[:8].to(accelerator.device)
    with torch.no_grad():
        final_loss = model(input_ids=first_global_batch, labels=first_global_batch).loss.item()

    if accelerator.is_main_process:
        result = {
            "losses": losses,
            "final_loss": final_loss,
            "learning_rates": learning_rates,
            "sample_ids": sample_ids,
            "world_size": accelerator.num_processes,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
