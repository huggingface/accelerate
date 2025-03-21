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
from utils import (
    parse_args,
    prepare_accelerate,
    prepare_torch,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LEARNING_RATE = 1e-4

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
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(**batch, use_cache=False)

        loss = outputs.loss
        losses.append(loss.item())
        accelerator.backward(loss)
        optimizer.step()

    return torch.tensor(losses)


def evaluate(args, config: dict, init_fn: Callable) -> torch.Tensor:
    model, optimizer, dataloader, accelerator = init_fn(args, config)
    loss = train(model, optimizer, dataloader, accelerator)
    return loss


def main():
    args = parse_args()
    evaluations = [
        functools.partial(evaluate, init_fn=functools.partial(prepare_torch, post_shard_optimizer=True)),
        functools.partial(
            evaluate, init_fn=functools.partial(prepare_torch, post_shard_optimizer=False, apply_optimizer_fix=True)
        ),
        functools.partial(
            evaluate, init_fn=functools.partial(prepare_torch, post_shard_optimizer=False, apply_optimizer_fix=False)
        ),
        functools.partial(evaluate, init_fn=prepare_accelerate),
    ]

    for evaluation in evaluations:
        results = evaluation(args, CONFIG)
        print(results)


if __name__ == "__main__":
    main()
