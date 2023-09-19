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

import inspect
import logging
import os
import random
import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.test_utils import execute_subprocess_async, require_cuda
from accelerate.utils import ProjectConfiguration, set_seed


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


def train(num_epochs, model, dataloader, optimizer, accelerator, scheduler=None):
    "Trains for `num_epochs`"
    rands = []
    for epoch in range(num_epochs):
        # Train quickly
        model.train()
        for batch in dataloader:
            x, y = batch
            outputs = model(x)
            loss = torch.nn.functional.mse_loss(outputs, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        rands.append(random.random())  # Introduce some randomness
        if scheduler is not None:
            scheduler.step()
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
    def test_with_save_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(total_limit=1, project_dir=tmpdir, automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state()

            # Save second state
            accelerator.save_state()
            self.assertEqual(len(os.listdir(accelerator.project_dir)), 1)

    def test_can_resume_training_with_folder(self):
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
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state(initial)
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            self.assertEqual(a, a2)
            self.assertEqual(b, b2)
            self.assertEqual(opt_state, opt_state2)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            checkpoint = os.path.join(tmpdir, "checkpoint")
            accelerator.save_state(checkpoint)

            # Load everything back in and make sure all states work
            accelerator.load_state(checkpoint)
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            self.assertEqual(a1, a3)
            self.assertEqual(b1, b3)
            self.assertEqual(opt_state1, opt_state3)
            self.assertEqual(ground_truth_rands, test_rands)

    def test_can_resume_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)

            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state()
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(iteration=1, automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            self.assertEqual(a, a2)
            self.assertEqual(b, b2)
            self.assertEqual(opt_state, opt_state2)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state()

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            self.assertEqual(a1, a3)
            self.assertEqual(b1, b3)
            self.assertEqual(opt_state1, opt_state3)
            self.assertEqual(ground_truth_rands, test_rands)

    def test_can_resume_training_checkpoints_relative_path(self):
        # See #1983
        # This test is like test_can_resume_training but uses a relative path for the checkpoint and automatically
        # infers the checkpoint path when loading.
        @contextmanager
        def temporary_relative_directory():
            # This is equivalent to tempfile.TemporaryDirectory() except that it returns a relative path
            rand_dir = f"test_path_{uuid.uuid4()}"
            os.mkdir(rand_dir)
            try:
                yield rand_dir
            finally:
                shutil.rmtree(rand_dir)

        with temporary_relative_directory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)

            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state()
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(iteration=1, automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state()  # <= infer the directory automatically
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            self.assertEqual(a, a2)
            self.assertEqual(b, b2)
            self.assertEqual(opt_state, opt_state2)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state()

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            self.assertEqual(a1, a3)
            self.assertEqual(b1, b3)
            self.assertEqual(opt_state1, opt_state3)
            self.assertEqual(ground_truth_rands, test_rands)

    def test_invalid_registration(self):
        t = torch.tensor([1, 2, 3])
        t1 = torch.tensor([2, 3, 4])
        net = DummyModel()
        opt = torch.optim.Adam(net.parameters())
        accelerator = Accelerator()
        with self.assertRaises(ValueError) as ve:
            accelerator.register_for_checkpointing(t, t1, net, opt)
        message = str(ve.exception)
        self.assertTrue("Item at index 0" in message)
        self.assertTrue("Item at index 1" in message)
        self.assertFalse("Item at index 2" in message)
        self.assertFalse("Item at index 3" in message)

    def test_with_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader, scheduler
            )
            # Save initial
            accelerator.save_state()
            scheduler_state = scheduler.state_dict()
            train(3, model, train_dataloader, optimizer, accelerator, scheduler)
            self.assertNotEqual(scheduler_state, scheduler.state_dict())

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            self.assertEqual(scheduler_state, scheduler.state_dict())

    def test_automatic_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader, scheduler
            )
            # Save initial
            accelerator.save_state()
            train(2, model, train_dataloader, optimizer, accelerator, scheduler)
            (a2, b2) = model.a.item(), model.b.item()
            # Save a first time
            accelerator.save_state()
            train(1, model, train_dataloader, optimizer, accelerator, scheduler)
            (a3, b3) = model.a.item(), model.b.item()

            # Load back in the last saved checkpoint, should point to a2, b2
            accelerator.load_state()
            self.assertNotEqual(a3, model.a.item())
            self.assertNotEqual(b3, model.b.item())
            self.assertEqual(a2, model.a.item())
            self.assertEqual(b2, model.b.item())

    def test_checkpoint_deletion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=2)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model = accelerator.prepare(model)
            # Save 3 states:
            for _ in range(11):
                accelerator.save_state()
            self.assertTrue(not os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_0")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_9")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_10")))

    @require_cuda
    def test_map_location(self):
        cmd = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]
        execute_subprocess_async(cmd, env=os.environ.copy())


if __name__ == "__main__":
    savedir = "/tmp/accelerate/state_checkpointing"
    model = DummyModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    train_dataloader, valid_dataloader = dummy_dataloaders()
    project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
    # Train baseline
    accelerator = Accelerator(project_dir=savedir, project_config=project_config, mixed_precision="no")
    if accelerator.process_index == 0:
        if os.path.exists(savedir):
            shutil.rmtree(savedir)
        os.makedirs(savedir)
    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, scheduler
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    train(3, model, train_dataloader, optimizer, accelerator, scheduler)
    # Check that the intial optimizer is loaded on the GPU
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == accelerator.device.type
    model = model.cpu()
    accelerator.wait_for_everyone()
    accelerator.save_state()
    accelerator.wait_for_everyone()

    # Check CPU state
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="cpu")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert (
        param_device.type == torch.device("cpu").type
    ), f"Loaded optimizer states did not match, expected to be loaded on the CPU but got {param_device}"

    # Check device state
    model.to(accelerator.device)
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="on_device")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert (
        param_device.type == accelerator.device.type
    ), f"Loaded optimizer states did not match, expected to be loaded on {accelerator.device} but got {param_device}"

    # Check error
    with pytest.raises(TypeError, match="Unsupported optimizer map location passed"):
        accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="invalid")
    accelerator.wait_for_everyone()
    if accelerator.process_index == 0:
        shutil.rmtree(savedir)
    accelerator.wait_for_everyone()
