# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import functools
from typing import Callable

import torch

from accelerate import Accelerator
from utils import parse_args, prepare_accelerate, prepare_torch


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LEARNING_RATE = 3e-5

CONFIG = {
    "model_name": MODEL_NAME,
    "learning_rate": LEARNING_RATE,
}


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
) -> torch.Tensor:
    losses = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch, use_cache=False)

        loss = outputs.loss
        losses.append(loss.item())
        accelerator.backward(loss)
        optimizer.step()

    return torch.tensor(losses)


def evaluate(args, config: dict, init_fn: Callable, run_name: str) -> torch.Tensor:
    model, optimizer, dataloader, accelerator, memory_tracker = init_fn(args, config)

    loss = train(model, optimizer, dataloader, accelerator)

    memory_tracker.stop()
    msg = f"""Results for {run_name} (rank 0):
Loss: {loss[-1].item()}
Peak Allocated Memory: {float(memory_tracker.peak_allocated_memory):.2f} MB
Peak Reserved Memory: {float(memory_tracker.peak_reserved_memory):.2f} MB
{"-" * 34}"""
    accelerator.print(msg)
    return loss


def main():
    args = parse_args()
    evaluations = [
        functools.partial(
            evaluate,
            init_fn=functools.partial(prepare_torch, post_shard_optimizer=False, apply_optimizer_fix=True),
            run_name="Optimizer Before FSDP (w/ fix)",
        ),
        functools.partial(
            evaluate,
            init_fn=functools.partial(prepare_torch, post_shard_optimizer=False, apply_optimizer_fix=False),
            run_name="Optimizer Before FSDP (w/o fix)",
        ),
        functools.partial(
            evaluate,
            init_fn=functools.partial(prepare_torch, post_shard_optimizer=True),
            run_name="Optimizer After FSDP",
        ),
        functools.partial(evaluate, init_fn=prepare_accelerate, run_name="Accelerate"),
    ]
    labels = [
        "Optimizer Before FSDP (w/ fix)",
        "Optimizer Before FSDP (w/o fix)",
        "Optimizer After FSDP",
        "Accelerate",
    ]

    results = {}

    for evaluation, label in zip(evaluations, labels):
        results[label] = evaluation(args, CONFIG)

    torch.testing.assert_close(
        results["Optimizer After FSDP"],
        results["Optimizer Before FSDP (w/ fix)"],
        msg="Optimizer After FSDP and Optimizer Before FSDP (w/ fix) should be the same",
    )

    torch.testing.assert_close(
        results["Optimizer After FSDP"],
        results["Accelerate"],
        msg="Optimizer After FSDP and Accelerate should be the same",
    )

    torch.testing.assert_close(
        results["Accelerate"],
        results["Optimizer Before FSDP (w/ fix)"],
        msg="Accelerate and Optimizer Before FSDP (w/ fix) should be the same",
    )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
