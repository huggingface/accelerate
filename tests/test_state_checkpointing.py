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
import uuid
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
from parameterized import parameterized_class
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    execute_subprocess_async,
    require_non_cpu,
    require_non_torch_xla,
    run_first,
)
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import (
    DistributedType,
    ProjectConfiguration,
    get_torch_accelerator_backend,
    patch_environment,
    set_seed,
)


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


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = "use_safetensors" if param["use_safetensors"] is True else "use_pytorch"
    return f"{func.__name__}_{param_based_name}"


@parameterized_class(("use_safetensors",), [[True], [False]], class_name_func=parameterized_custom_name_func)
class CheckpointTest(AccelerateTestCase):
    def test_rng_state_restored(self):
        accelerator = Accelerator()
        devices = [accelerator.device]
        device_type, device_module = get_torch_accelerator_backend()
        # Unified backends and MPS save only the current device's RNG state.
        if device_type not in (None, "mps") and device_module is not torch.accelerator:
            devices = [torch.device(device_type, i) for i in range(device_module.device_count())]
        with tempfile.TemporaryDirectory() as tmpdir:
            accelerator.save_state(tmpdir, safe_serialization=self.use_safetensors)
            expected_cpu = torch.rand(8)
            expected_devices = [torch.rand(8, device=device) for device in devices]

            accelerator.load_state(tmpdir)
            torch.testing.assert_close(torch.rand(8), expected_cpu, rtol=0, atol=0)
            for device, expected in zip(devices, expected_devices):
                torch.testing.assert_close(torch.rand(8, device=device), expected, rtol=0, atol=0)

    def check_adam_state(self, state1, state2, distributed_type):
        # For DistributedType.XLA, the `accelerator.save_state` function calls `xm._maybe_convert_to_cpu` before saving.
        # As a result, all tuple values are converted to lists. Therefore, we need to convert them back here.
        # Remove this code once Torch XLA fixes this issue.
        if distributed_type == DistributedType.XLA:
            state1["param_groups"][0]["betas"] = tuple(state1["param_groups"][0]["betas"])
            state2["param_groups"][0]["betas"] = tuple(state2["param_groups"][0]["betas"])
        assert state1 == state2

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
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Save second state
            accelerator.save_state(safe_serialization=self.use_safetensors)
            assert len(os.listdir(accelerator.project_dir)) == 1

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
            accelerator.save_state(initial, safe_serialization=self.use_safetensors)
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
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            checkpoint = os.path.join(tmpdir, "checkpoint")
            accelerator.save_state(checkpoint, safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(checkpoint)
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

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
            accelerator.save_state(safe_serialization=self.use_safetensors)
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
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

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
            accelerator.save_state(safe_serialization=self.use_safetensors)
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
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)
            assert opt_state == opt_state2

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

    def test_invalid_registration(self):
        t = torch.tensor([1, 2, 3])
        t1 = torch.tensor([2, 3, 4])
        net = DummyModel()
        opt = torch.optim.Adam(net.parameters())
        accelerator = Accelerator()
        with self.assertRaises(ValueError) as ve:
            accelerator.register_for_checkpointing(t, t1, net, opt)
        message = str(ve.exception)
        assert "Item at index 0" in message
        assert "Item at index 1" in message
        assert "Item at index 2" not in message
        assert "Item at index 3" not in message

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
            accelerator.save_state(safe_serialization=self.use_safetensors)
            scheduler_state = scheduler.state_dict()
            train(3, model, train_dataloader, optimizer, accelerator, scheduler)
            assert scheduler_state != scheduler.state_dict()

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            assert scheduler_state == scheduler.state_dict()

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
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train(2, model, train_dataloader, optimizer, accelerator, scheduler)
            (a2, b2) = model.a.item(), model.b.item()
            # Save a first time
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train(1, model, train_dataloader, optimizer, accelerator, scheduler)
            (a3, b3) = model.a.item(), model.b.item()

            # Load back in the last saved checkpoint, should point to a2, b2
            accelerator.load_state()
            assert a3 != model.a.item()
            assert b3 != model.b.item()
            assert a2 == model.a.item()
            assert b2 == model.b.item()

    def test_automatic_loading_skips_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save checkpoint_0 and record its weights
            accelerator.save_state(safe_serialization=self.use_safetensors)
            (a0, b0) = model.a.item(), model.b.item()
            train(2, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            # Save checkpoint_1, then simulate a torn write by truncating its model file
            accelerator.save_state(safe_serialization=self.use_safetensors)
            model_file = "model.safetensors" if self.use_safetensors else "pytorch_model.bin"
            newest = os.path.join(tmpdir, "checkpoints", "checkpoint_1", model_file)
            with open(newest, "r+b") as f:
                f.truncate(os.path.getsize(newest) // 2)

            # Loading with no path should skip the torn checkpoint_1 and load checkpoint_0
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state()
            assert model.a.item() == a0
            assert model.b.item() == b0
            assert model.a.item() != a1
            assert model.b.item() != b1

            # If every checkpoint is incomplete, the load should still raise
            oldest = os.path.join(tmpdir, "checkpoints", "checkpoint_0", model_file)
            with open(oldest, "r+b") as f:
                f.truncate(os.path.getsize(oldest) // 2)
            with self.assertRaises(Exception):
                accelerator.load_state()

    def test_crash_during_save_leaves_no_partial_checkpoint(self):
        # Regression test for #4549 (silent case): a crash just before `random_states_0.pkl` is written leaves a
        # folder that loads without an error but without the random states, so the fallback in `load_state()` never
        # runs. With the commit point in `save_state()`, the interrupted save must never produce a `checkpoint_<i>`
        # folder, and the restart must resume the previous complete checkpoint with its random states.
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            project_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            def train_step():
                accelerator.backward(model(torch.randn(8, 4)).pow(2).mean())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # checkpoint_0 (step 1) and checkpoint_1 (step 2)
            train_step()
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train_step()
            accelerator.save_state(safe_serialization=self.use_safetensors)
            ckpt1_weights = (model.weight.detach().clone(), model.bias.detach().clone())
            ckpt1_rng = torch.get_rng_state().clone()
            assert scheduler.scheduler.last_epoch == 2
            # Move the RNG state forward so a load that skips the random states is detectable
            torch.randn(16)

            # Step 3, then crash just before `random_states_0.pkl` is written
            train_step()
            assert scheduler.scheduler.last_epoch == 3
            real_save = torch.save

            def crashing_save(obj, f, *args, **kwargs):
                if str(f).endswith("random_states_0.pkl"):
                    raise RuntimeError("simulated crash during save_state")
                return real_save(obj, f, *args, **kwargs)

            with mock.patch("torch.save", side_effect=crashing_save), self.assertRaises(RuntimeError):
                accelerator.save_state(safe_serialization=self.use_safetensors)

            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            # No partial `checkpoint_2` folder may exist: the interrupted save only leaves its staging directory.
            assert not os.path.exists(os.path.join(checkpoints_dir, "checkpoint_2"))
            assert os.path.isdir(os.path.join(checkpoints_dir, ".checkpoint_2.tmp"))

            # Restart: load_state() must resume checkpoint_1, with its random states restored
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 2
            torch.testing.assert_close(model.weight.detach(), ckpt1_weights[0])
            torch.testing.assert_close(model.bias.detach(), ckpt1_weights[1])
            assert torch.equal(torch.get_rng_state(), ckpt1_rng)

            # Saving again after the crashed save reuses the same iteration: the leftover staging directory is
            # removed first and the new checkpoint_2 is committed.
            accelerator.save_state(safe_serialization=self.use_safetensors)
            assert os.path.isdir(os.path.join(checkpoints_dir, "checkpoint_2"))
            assert not os.path.exists(os.path.join(checkpoints_dir, ".checkpoint_2.tmp"))

            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 3

    def test_crash_during_save_with_total_limit_1_keeps_previous_checkpoint(self):
        # Regression test for #4549 (`total_limit=1` case): retention used to delete the previous checkpoint before
        # the new save started, so a crash during the save left only the partial folder and every restart failed.
        # Retention now runs only after a save is committed, so the previous checkpoint must survive the crash.
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            project_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=1)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            def train_step():
                accelerator.backward(model(torch.randn(8, 4)).pow(2).mean())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # checkpoint_0 (step 1) and checkpoint_1 (step 2); total_limit=1 keeps only checkpoint_1
            train_step()
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train_step()
            accelerator.save_state(safe_serialization=self.use_safetensors)
            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            assert sorted(os.listdir(checkpoints_dir)) == ["checkpoint_1"]
            ckpt1_weights = (model.weight.detach().clone(), model.bias.detach().clone())
            assert scheduler.scheduler.last_epoch == 2

            # Step 3, then crash while writing `optimizer.bin` of checkpoint_2
            train_step()
            assert scheduler.scheduler.last_epoch == 3
            real_save = torch.save

            def crashing_save(obj, f, *args, **kwargs):
                if str(f).endswith("optimizer.bin"):
                    raise RuntimeError("simulated crash during save_state")
                return real_save(obj, f, *args, **kwargs)

            with mock.patch("torch.save", side_effect=crashing_save), self.assertRaises(RuntimeError):
                accelerator.save_state(safe_serialization=self.use_safetensors)

            # The previous checkpoint must survive, and no partial checkpoint_2 may exist
            assert os.path.isdir(os.path.join(checkpoints_dir, "checkpoint_1"))
            assert not os.path.exists(os.path.join(checkpoints_dir, "checkpoint_2"))

            # Restart: load_state() must resume checkpoint_1 instead of failing
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 2
            torch.testing.assert_close(model.weight.detach(), ckpt1_weights[0])
            torch.testing.assert_close(model.bias.detach(), ckpt1_weights[1])

    def test_restart_restores_checkpoint_iteration_with_total_limit(self):
        # Regression test for the follow-up to #4595 reported on #4597: `load_state()` did not restore
        # `project_configuration.iteration`, so after a restart the next saves restarted numbering at
        # `checkpoint_0` and, with `total_limit` set, retention deleted each new checkpoint as soon as it
        # was committed, so training could never advance past the resumed step. A fresh
        # `ProjectConfiguration` is used for each restart to simulate a new process (iteration starts at 0).
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            def train_step():
                accelerator.backward(model(torch.randn(8, 4)).pow(2).mean())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Steps 1-4; total_limit=2 keeps only checkpoint_2 (step 3) and checkpoint_3 (step 4)
            for _ in range(4):
                train_step()
                accelerator.save_state(safe_serialization=self.use_safetensors)
            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            assert sorted(os.listdir(checkpoints_dir)) == ["checkpoint_2", "checkpoint_3"]
            assert scheduler.scheduler.last_epoch == 4
            ckpt3_weights = (model.weight.detach().clone(), model.bias.detach().clone())

            # Restart: load_state() must resume step 4 and restore the counter to one past the highest
            # existing checkpoint (4), so the next saves are checkpoint_4 and checkpoint_5.
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=restart_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 4
            torch.testing.assert_close(model.weight.detach(), ckpt3_weights[0])
            torch.testing.assert_close(model.bias.detach(), ckpt3_weights[1])
            assert restart_accelerator.project_configuration.iteration == 4

            # Steps 5 and 6 must be kept (not pruned right after being committed)
            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            assert os.path.isdir(os.path.join(checkpoints_dir, "checkpoint_4"))
            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            assert sorted(os.listdir(checkpoints_dir)) == ["checkpoint_4", "checkpoint_5"]
            assert scheduler.scheduler.last_epoch == 6

            # A further restart must resume step 6, not step 4
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            verify_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            verify_accelerator = Accelerator(project_dir=tmpdir, project_config=verify_config)
            model, optimizer, scheduler = verify_accelerator.prepare(model, optimizer, scheduler)
            verify_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 6
            assert verify_accelerator.project_configuration.iteration == 6

    def test_explicit_load_state_restores_checkpoint_iteration_with_total_limit(self):
        # Regression test for the follow-up reported on #4597: after an explicit
        # `load_state(<path>)` of a checkpoint from the project's `checkpoints/` folder,
        # `project_configuration.iteration` stayed at 0, so the next saves restarted numbering at
        # `checkpoint_0` and, with `total_limit` set, retention deleted each new checkpoint as soon as
        # it was committed. A fresh `ProjectConfiguration` is used for each restart to simulate a new
        # process (iteration starts at 0).
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            def train_step():
                accelerator.backward(model(torch.randn(8, 4)).pow(2).mean())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Steps 1-4; total_limit=2 keeps only checkpoint_2 (step 3) and checkpoint_3 (step 4)
            for _ in range(4):
                train_step()
                accelerator.save_state(safe_serialization=self.use_safetensors)
            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            assert sorted(os.listdir(checkpoints_dir)) == ["checkpoint_2", "checkpoint_3"]
            assert scheduler.scheduler.last_epoch == 4
            ckpt3_weights = (model.weight.detach().clone(), model.bias.detach().clone())

            # Restart with an explicit load of the newest checkpoint: the counter must be restored to
            # one past the highest existing checkpoint (4), so the next saves are checkpoint_4 and
            # checkpoint_5.
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=restart_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state(os.path.join(checkpoints_dir, "checkpoint_3"))
            assert scheduler.scheduler.last_epoch == 4
            torch.testing.assert_close(model.weight.detach(), ckpt3_weights[0])
            torch.testing.assert_close(model.bias.detach(), ckpt3_weights[1])
            assert restart_accelerator.project_configuration.iteration == 4

            # Steps 5 and 6 must be kept (not pruned right after being committed)
            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            assert os.path.isdir(os.path.join(checkpoints_dir, "checkpoint_4"))
            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            assert sorted(os.listdir(checkpoints_dir)) == ["checkpoint_4", "checkpoint_5"]
            assert scheduler.scheduler.last_epoch == 6

            # A further restart must resume step 6, not step 4
            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            verify_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True, total_limit=2)
            verify_accelerator = Accelerator(project_dir=tmpdir, project_config=verify_config)
            model, optimizer, scheduler = verify_accelerator.prepare(model, optimizer, scheduler)
            verify_accelerator.load_state()
            assert scheduler.scheduler.last_epoch == 6
            assert verify_accelerator.project_configuration.iteration == 6

    def test_explicit_load_state_restores_checkpoint_iteration_without_total_limit(self):
        # Same scenario as above but without `total_limit`: after an explicit load of the newest
        # checkpoint, the next saves must continue at `checkpoint_4`/`checkpoint_5` instead of raising
        # `checkpoint_0 ... already exists`.
        with tempfile.TemporaryDirectory() as tmpdir:
            project_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True)
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

            def train_step():
                accelerator.backward(model(torch.randn(8, 4)).pow(2).mean())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for _ in range(4):
                train_step()
                accelerator.save_state(safe_serialization=self.use_safetensors)
            checkpoints_dir = os.path.join(tmpdir, "checkpoints")
            assert sorted(os.listdir(checkpoints_dir)) == [
                "checkpoint_0",
                "checkpoint_1",
                "checkpoint_2",
                "checkpoint_3",
            ]

            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=restart_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state(os.path.join(checkpoints_dir, "checkpoint_3"))
            assert scheduler.scheduler.last_epoch == 4
            assert restart_accelerator.project_configuration.iteration == 4

            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            train_step()
            restart_accelerator.save_state(safe_serialization=self.use_safetensors)
            assert sorted(os.listdir(checkpoints_dir)) == [
                "checkpoint_0",
                "checkpoint_1",
                "checkpoint_2",
                "checkpoint_3",
                "checkpoint_4",
                "checkpoint_5",
            ]
            assert scheduler.scheduler.last_epoch == 6

    def test_explicit_load_state_outside_checkpoints_leaves_iteration_unchanged(self):
        # An explicit load from a folder outside the project's `checkpoints/` folder must not touch
        # the automatic checkpoint counter, even with automatic checkpoint naming enabled. The external
        # checkpoint is saved with a separate accelerator that has automatic naming off, since an
        # explicit `output_dir` is ignored when automatic naming is on.
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            save_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=False)
            accelerator = Accelerator(project_dir=tmpdir, project_config=save_config)
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            accelerator.save_state(os.path.join(tmpdir, "my_checkpoint"), safe_serialization=self.use_safetensors)
            assert os.path.isdir(os.path.join(tmpdir, "my_checkpoint"))

            set_seed(123)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            restart_config = ProjectConfiguration(project_dir=tmpdir, automatic_checkpoint_naming=True)
            restart_accelerator = Accelerator(project_dir=tmpdir, project_config=restart_config)
            model, optimizer, scheduler = restart_accelerator.prepare(model, optimizer, scheduler)
            restart_accelerator.load_state(os.path.join(tmpdir, "my_checkpoint"))
            assert scheduler.scheduler.last_epoch == 0
            assert restart_accelerator.project_configuration.iteration == 0

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
                accelerator.save_state(safe_serialization=self.use_safetensors)
            assert not os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_9"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_10"))

    @run_first
    @require_non_cpu
    @require_non_torch_xla
    def test_map_location(self):
        cmd = DEFAULT_LAUNCH_COMMAND + [inspect.getfile(self.__class__)]

        env_kwargs = dict(use_safe_tensors=str(self.use_safetensors), omp_num_threads="1")
        with patch_environment(**env_kwargs):
            execute_subprocess_async(cmd)


if __name__ == "__main__":
    use_safetensors = os.environ.get("USE_SAFETENSORS", "False") == "True"
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
    # Check that the initial optimizer is loaded on the GPU
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == accelerator.device.type
    model = model.cpu()
    accelerator.wait_for_everyone()
    accelerator.save_state(safe_serialization=use_safetensors)
    accelerator.wait_for_everyone()

    # Check CPU state
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="cpu")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == torch.device("cpu").type, (
        f"Loaded optimizer states did not match, expected to be loaded on the CPU but got {param_device}"
    )

    # Check device state
    model.to(accelerator.device)
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="on_device")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == accelerator.device.type, (
        f"Loaded optimizer states did not match, expected to be loaded on {accelerator.device} but got {param_device}"
    )

    # Check error
    with pytest.raises(TypeError, match="Unsupported optimizer map location passed"):
        accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="invalid")
    accelerator.wait_for_everyone()
    if accelerator.process_index == 0:
        shutil.rmtree(savedir)
    accelerator.wait_for_everyone()
