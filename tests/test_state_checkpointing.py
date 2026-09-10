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

import pytest
import torch
from parameterized import parameterized, parameterized_class
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
    DataLoaderConfiguration,
    DistributedType,
    ProjectConfiguration,
    is_torchdata_stateful_dataloader_available,
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

    # No default-sampler case beyond the epoch boundary: in a single process it draws from the global RNG (or, with
    # `use_stateful_dataloader`, from torchdata's own generator), which `load_state` restores as of the checkpoint, so
    # the interrupted epoch is redrawn. `test_script.py` covers the multi-process default sampler, where `prepare`
    # attaches a private generator.
    @parameterized.expand(
        [(True, False, False), (True, True, False), (False, False, False), (True, True, True)],
        name_func=lambda f, n, p: f"{f.__name__}_seedable_{p.args[0]}_mid_epoch_{p.args[1]}_stateful_{p.args[2]}",
    )
    @require_non_torch_xla
    def test_resume_restores_dataloader_shuffle_order(self, use_seedable_sampler, mid_epoch, use_stateful_dataloader):
        """
        After `load_state`, a shuffled dataloader must continue with the order it would have produced had training not
        been interrupted, not replay epoch 0 (https://github.com/huggingface/accelerate/issues/3996). A checkpoint
        taken inside an epoch is resumed with `skip_first_batches` (a stateful dataloader is positioned by `load_state`
        itself), and the epoch after that one must follow too.
        """
        if use_stateful_dataloader and not is_torchdata_stateful_dataloader_available():
            self.skipTest("torchdata.stateful_dataloader is not available")
        skip_batches, batch_size = 3, 4

        def make_dataloader(accelerator):
            dataloader = DataLoader(TensorDataset(torch.arange(32)), batch_size=batch_size, shuffle=True)
            return accelerator.prepare(dataloader)

        def epoch_order(dataloader):
            return [sample for batch in dataloader for sample in batch[0].tolist()]

        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            dataloader_config = DataLoaderConfiguration(
                use_seedable_sampler=use_seedable_sampler, use_stateful_dataloader=use_stateful_dataloader
            )
            accelerator = Accelerator(dataloader_config=dataloader_config)
            train_dataloader = make_dataloader(accelerator)
            orders = [epoch_order(train_dataloader) for _ in range(2)]
            if mid_epoch:
                iterator = iter(train_dataloader)
                for _ in range(skip_batches):
                    next(iterator)
                accelerator.save_state(tmpdir, safe_serialization=self.use_safetensors)
                expected = [[sample for batch in iterator for sample in batch[0].tolist()]]
            else:
                accelerator.save_state(tmpdir, safe_serialization=self.use_safetensors)
                expected = [epoch_order(train_dataloader)]
            expected.append(epoch_order(train_dataloader))
            assert expected[-1] != orders[0], "The dataset is too small to tell epochs apart"

            # Resume from scratch, as a new script run would
            set_seed(42)
            accelerator = Accelerator(dataloader_config=dataloader_config)
            train_dataloader = make_dataloader(accelerator)
            accelerator.load_state(tmpdir)
            if mid_epoch and not use_stateful_dataloader:
                resumed = [epoch_order(accelerator.skip_first_batches(train_dataloader, skip_batches))]
            else:
                resumed = [epoch_order(train_dataloader)]
            resumed.append(epoch_order(train_dataloader))

            assert resumed[0] != orders[0][skip_batches * batch_size if mid_epoch else 0 :], (
                "Resumed dataloader replayed the epoch-0 shuffle order"
            )
            assert resumed == expected

    @parameterized.expand([(True,), (False,)], name_func=lambda f, n, p: f"{f.__name__}_seedable_{p.args[0]}")
    @require_non_torch_xla
    def test_resume_after_checkpoint_on_last_batch(self, use_seedable_sampler):
        """
        A checkpoint taken while processing the last batch of an epoch (what `checkpointing_steps` produces whenever it
        divides the number of batches) must resume with the next epoch, not replay the one that just finished.
        """

        def make_dataloader(accelerator):
            dataloader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True)
            return accelerator.prepare(dataloader)

        def epoch_order(dataloader):
            return [sample for batch in dataloader for sample in batch[0].tolist()]

        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            dataloader_config = DataLoaderConfiguration(use_seedable_sampler=use_seedable_sampler)
            accelerator = Accelerator(dataloader_config=dataloader_config)
            train_dataloader = make_dataloader(accelerator)
            epoch_order(train_dataloader)
            just_finished = []
            for step, batch in enumerate(train_dataloader):
                just_finished += batch[0].tolist()
                if step == len(train_dataloader) - 1:
                    accelerator.save_state(tmpdir, safe_serialization=self.use_safetensors)
            expected_next_epoch = epoch_order(train_dataloader)

            set_seed(42)
            accelerator = Accelerator(dataloader_config=dataloader_config)
            train_dataloader = make_dataloader(accelerator)
            accelerator.load_state(tmpdir)
            # The example computes a resume step of 0 for this checkpoint and still goes through `skip_first_batches`
            resumed = epoch_order(accelerator.skip_first_batches(train_dataloader, 0))

            assert resumed != just_finished, "Resumed dataloader replayed the epoch the checkpoint closed"
            assert resumed == expected_next_epoch

    @require_non_torch_xla
    def test_resume_restores_user_generator(self):
        "A `generator` passed to the `DataLoader` is restored by `load_state`, so a resumed run keeps its shuffle order."

        def make_dataloader(accelerator):
            generator = torch.Generator().manual_seed(0)
            dataloader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True, generator=generator)
            return accelerator.prepare(dataloader)

        def epoch_order(dataloader):
            return [sample for batch in dataloader for sample in batch[0].tolist()]

        with tempfile.TemporaryDirectory() as tmpdir:
            accelerator = Accelerator()
            train_dataloader = make_dataloader(accelerator)
            first_epoch = epoch_order(train_dataloader)
            epoch_order(train_dataloader)
            accelerator.save_state(tmpdir, safe_serialization=self.use_safetensors)
            expected_continuation = epoch_order(train_dataloader)

            accelerator = Accelerator()
            train_dataloader = make_dataloader(accelerator)
            accelerator.load_state(tmpdir)
            resumed = epoch_order(train_dataloader)

            assert resumed != first_epoch, "Resumed dataloader replayed the epoch-0 shuffle order"
            assert resumed == expected_continuation

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
