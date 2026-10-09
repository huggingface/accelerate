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
import itertools
import os
import sys
import tempfile
from copy import deepcopy
from unittest.mock import patch

import torch
from parameterized import parameterized

from accelerate import Accelerator
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
    require_deepspeed,
)
from accelerate.utils import DeepSpeedPlugin, DistributedType
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler


@require_deepspeed
class DeepSpeedSchedulerConfigTest(AccelerateTestCase):
    def test_checkpoint_with_new_engine(self):
        script = path_in_accelerate_package(
            "test_utils", "scripts", "external_deps", "test_ds_scheduler_checkpoint.py"
        )
        env = os.environ.copy()
        env.setdefault("DS_ACCELERATOR", "cuda" if torch.cuda.is_available() else "cpu")
        env.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(get_torch_dist_unique_port()),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                script,
                "--checkpoint_dir",
                directory,
            ]
            self.assertEqual(execute_subprocess_async(command, env=env).returncode, 0)
            # A separate worker reconstructs the optimizer and scheduler before loading the checkpoint.
            self.assertEqual(execute_subprocess_async(command + ["--resume"], env=env).returncode, 0)

    @parameterized.expand(
        list(itertools.product(["WarmupLR", "WarmupDecayLR"], ["dummy", "custom"], ["auto", "scalar", "list"]))
    )
    def test_parameter_group_learning_rates(self, scheduler_name, optimizer_type, max_lr_type):
        import deepspeed

        max_lr = {"auto": "auto", "scalar": 0.003, "list": [0.03, 0.0002]}[max_lr_type]
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
            "scheduler": {
                "type": scheduler_name,
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": max_lr,
                    "warmup_num_steps": "auto",
                    "warmup_type": "linear",
                },
            },
        }
        if scheduler_name == "WarmupDecayLR":
            config["scheduler"]["params"]["total_num_steps"] = "auto"
        if optimizer_type == "dummy":
            config["optimizer"] = {"type": "Adam", "params": {"lr": "auto"}}

        model = torch.nn.Linear(2, 2)

        def make_optimizer():
            return torch.optim.SGD(
                [{"params": [model.weight], "lr": 0.02}, {"params": [model.bias], "lr": 1e-5}], lr=1e-5
            )

        optimizer = DummyOptim(model.parameters(), lr=1e-5) if optimizer_type == "dummy" else make_optimizer()
        scheduler = DummyScheduler(optimizer, warmup_num_steps=4, total_num_steps=20)
        accelerator = Accelerator(cpu=True)
        accelerator.state.distributed_type = DistributedType.DEEPSPEED
        accelerator.state.deepspeed_plugins = DeepSpeedPlugin(hf_ds_config=config)
        accelerator.deepspeed_engine_wrapped = None

        def initialize(model, config_params, optimizer=None, model_parameters=None, lr_scheduler=None):
            # Only engine initialization is replaced: optimizer groups can appear after DummyOptim is resolved.
            # Exercise the actual Accelerate preparation path and actual DeepSpeed scheduler on CPU.
            if optimizer is None:
                optimizer = make_optimizer()
            if lr_scheduler is not None:
                lr_scheduler = lr_scheduler(optimizer)
            else:
                scheduler_config = config_params["scheduler"]
                lr_scheduler = getattr(deepspeed.runtime.lr_schedules, scheduler_config["type"])(
                    optimizer, **scheduler_config["params"]
                )
            model.lr_scheduler = lr_scheduler
            return model, optimizer, None, lr_scheduler

        with patch("deepspeed.initialize", side_effect=initialize):
            _, optimizer, scheduler = accelerator._prepare_deepspeed(model, optimizer, scheduler)

        scheduler = scheduler.scheduler
        expected_max = {"auto": [0.02, 1e-5], "scalar": [0.003, 0.003], "list": [0.03, 0.0002]}[max_lr_type]
        self.assertEqual(scheduler.max_lrs, expected_max)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0, 0])
        if max_lr_type == "auto":
            self.assertIs(model.lr_scheduler, scheduler)
        expected_config = expected_max if max_lr_type == "auto" else max_lr
        self.assertEqual(accelerator.deepspeed_config["scheduler"]["params"]["warmup_max_lr"], expected_config)
        for step in [1, 4, 10]:
            scheduler.step(step)
            factor = step / 4 if step < 4 else (20 - step) / 16 if scheduler_name == "WarmupDecayLR" else 1
            self.assertEqual([group["lr"] for group in optimizer.param_groups], [lr * factor for lr in expected_max])

        optimizer_state = deepcopy(optimizer.state_dict())
        scheduler_state = deepcopy(scheduler.state_dict())
        scheduler.step()
        expected_next = scheduler.get_last_lr()
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), expected_next)

    @parameterized.expand(["WarmupLR", "WarmupDecayLR"])
    def test_optimizer_rebuilt_by_zero(self, scheduler_name):
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
            "optimizer": {"type": "Adagrad", "params": {"lr": "auto"}},
            "scheduler": {
                "type": scheduler_name,
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "warmup_type": "linear",
                },
            },
        }
        if scheduler_name == "WarmupDecayLR":
            config["scheduler"]["params"]["total_num_steps"] = "auto"
        model = torch.nn.Linear(2, 2)
        optimizer = DummyOptim(model.parameters(), lr=0.02)
        scheduler = DummyScheduler(optimizer, warmup_num_steps=4, total_num_steps=20)
        accelerator = Accelerator(cpu=True)
        accelerator.state.distributed_type = DistributedType.DEEPSPEED
        accelerator.state.deepspeed_plugins = DeepSpeedPlugin(hf_ds_config=config)
        accelerator.deepspeed_engine_wrapped = None

        def initialize(model, config_params, model_parameters, lr_scheduler):
            basic_optimizer = torch.optim.Adagrad(model_parameters, lr=0.02)
            # Run the real ZeRO state-initialization method: Adagrad must be recreated for the flat parameters.
            zero_optimizer = object.__new__(DeepSpeedZeroOptimizer)
            zero_optimizer.optimizer = basic_optimizer
            zero_optimizer.bit16_groups = [list(model.parameters())]
            zero_optimizer.single_partition_of_fp32_groups = [torch.nn.Parameter(torch.ones(6))]
            zero_optimizer.partition_size = [6]
            zero_optimizer.device = torch.device("cpu")
            zero_optimizer.cpu_offload = False
            zero_optimizer.cpu_offload_pin_memory = False
            zero_optimizer.initialize_optimizer_states()
            self.assertIsNot(zero_optimizer.optimizer, basic_optimizer)
            # This is the object DeepSpeed passes to client scheduler factories, before returning its wrapper.
            scheduler = lr_scheduler(basic_optimizer)
            model.lr_scheduler = scheduler
            return model, zero_optimizer, None, scheduler

        with patch("deepspeed.initialize", side_effect=initialize):
            model, optimizer, scheduler = accelerator._prepare_deepspeed(model, optimizer, scheduler)

        self.assertIs(scheduler.scheduler.optimizer, optimizer.optimizer.optimizer)
        self.assertIs(model.lr_scheduler, scheduler.scheduler)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0])
        scheduler.scheduler.step(2)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.01])
