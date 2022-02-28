# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import os
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.utils import set_seed


def dummy_dataloaders(a=2, b=3, batch_size=16, n_train_batches: int = 10, n_valid_batches: int = 2):
    "Generates a tuple of dummy DataLoaders to test with"

    def get_dataset(n_batches):
        x = torch.randn(batch_size * n_batches, 1)
        return TensorDataset(x, a * x + b + 0.1 * torch.randn(batch_size * n_batches, 1))

    train_dataset = get_dataset(n_train_batches)
    valid_dataset = get_dataset(n_valid_batches)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    return (train_dataloader, valid_dataloader)


class DummyModel(nn.Module):
    "Simple model to do y=mx+b"

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


def test_can_resume_training(args):
    with tempfile.TemporaryDirectory() as tmpdir:
        set_seed(42)
        model = DummyModel()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        train_dataloader, valid_dataloader = dummy_dataloaders()
        # Train baseline
        accelerator = Accelerator(
            fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision, device_placement=True
        )
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
        # Save initial
        initial = os.path.join(tmpdir, "initial")
        accelerator.save_state(initial)
        model_unwrapped = accelerator.unwrap_model(model)
        (a, b) = model_unwrapped.a.item(), model_unwrapped.b.item()
        opt_state = optimizer.state_dict()
        for epoch in range(3):
            # Train quickly
            model.train()
            for step, batch in enumerate(train_dataloader):
                x, y = batch
                outputs = model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            random.random()  # For each random num
        model_unwrapped = accelerator.unwrap_model(model)
        (a1, b1) = model_unwrapped.a.item(), model_unwrapped.b.item()
        opt_state1 = optimizer.state_dict()

        # Train partially
        accelerator.load_state(initial)
        model_unwrapped = accelerator.unwrap_model(model)
        (a2, b2) = model_unwrapped.a.item(), model_unwrapped.b.item()
        opt_state2 = optimizer.state_dict()
        assert a == a2
        assert b == b2
        assert opt_state == opt_state2
        # Reset seeed
        set_seed(42)
        # Rebuild the dataloaders again here
        # Ensure all numbers align

        for epoch in range(2):
            # Train quickly
            model.train()
            for step, batch in enumerate(train_dataloader):
                x, y = batch
                outputs = model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        # Save everything
        checkpoint = os.path.join(tmpdir, "checkpoint")
        accelerator.save_state(checkpoint)

        # Load everything back in and make sure all states work
        accelerator.load_state(checkpoint)
        for epoch in range(1):
            # Train quickly
            model.train()
            for step, batch in enumerate(train_dataloader):
                x, y = batch
                outputs = model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        model_unwrapped = accelerator.unwrap_model(model)
        (a3, b3) = model_unwrapped.a.item(), model_unwrapped.b.item()
        opt_state3 = optimizer.state_dict()
        assert a1 == a3
        assert b1 == b3
        assert opt_state1 == opt_state3


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    test_can_resume_training(args)


if __name__ == "__main__":
    main()
