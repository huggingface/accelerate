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

import logging
import os
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.utils import set_seed


logger = logging.getLogger(__name__)


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


def train(num_epochs, model, dataloader, optimizer, accelerator):
    "Trains for `num_epochs`"
    rands = []
    for epoch in range(num_epochs):
        # Train quickly
        model.train()
        for step, batch in enumerate(dataloader):
            x, y = batch
            outputs = model(x)
            loss = torch.nn.functional.mse_loss(outputs, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        rands.append(random.random())  # Introduce some randomness
    return rands


class DummyModel(nn.Module):
    "Simple model to do y=mx+b"

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


class CheckpointTest(unittest.TestCase):
    def test_can_resume_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            # Train baseline
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            initial = os.path.join(tmpdir, "initial")
            accelerator.save_state(initial)
            model_unwrapped = accelerator.unwrap_model(model)
            (a, b) = model_unwrapped.a.item(), model_unwrapped.b.item()
            opt_state = optimizer.state_dict()

            # Train partially
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            rand_a = [random.getstate(), np.random.get_state(), torch.get_rng_state()]
            if torch.cuda.is_available:
                rand_a += [torch.cuda.get_rng_state()]
            model_unwrapped = accelerator.unwrap_model(model)
            (a1, b1) = model_unwrapped.a.item(), model_unwrapped.b.item()
            opt_state1 = optimizer.state_dict()

            # Test `load_state`
            accelerator.load_state(initial)
            model_unwrapped = accelerator.unwrap_model(model)
            (a2, b2) = model_unwrapped.a.item(), model_unwrapped.b.item()
            opt_state2 = optimizer.state_dict()
            self.assertEqual(a, a2)
            self.assertEqual(b, b2)
            self.assertEqual(opt_state, opt_state2)

            set_seed(42)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader, optimizer = accelerator.prepare(
                train_dataloader, valid_dataloader, optimizer
            )
            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            checkpoint = os.path.join(tmpdir, "checkpoint")
            accelerator.save_state(checkpoint)
            # Load everything back in and make sure all states work
            accelerator.load_state(checkpoint)
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            rand_b = [random.getstate(), np.random.get_state(), torch.get_rng_state()]
            if torch.cuda.is_available:
                rand_b += [torch.cuda.get_rng_state()]
            model_unwrapped = accelerator.unwrap_model(model)
            (a3, b3) = model_unwrapped.a.item(), model_unwrapped.b.item()
            opt_state3 = optimizer.state_dict()
            logger.info(f"Model (a) states: \n\t{a}\n\t{a1}\n\t{a2}\n\t{a3}")
            logger.info(f"Model (b) states: \n\t{b}\n\t{b1}\n\t{b2}\n\t{b3}")
            logger.info(f"Optimizer states: \n\t{opt_state}\n\t{opt_state1}\n\t{opt_state2}\n\t{opt_state3}")
            logger.info(f"Rands states: \n\t{ground_truth_rands}\n\t{test_rands}")
            for i, (a, b) in enumerate(zip(rand_a, rand_b)):
                for c, d in zip(a, b):
                    if isinstance(c, (list, np.ndarray)):
                        self.assertTrue(all([e == f for e, f in zip(c, d)]))
                        logger.info(f"Passed: {i}")
                    else:
                        self.assertEqual(c, d)
                        logger.info(f"Passed: {i}")
            self.assertEqual(a1, a3)
            self.assertEqual(b1, b3)
            self.assertEqual(opt_state1, opt_state3)
            self.assertEqual(ground_truth_rands, test_rands)
