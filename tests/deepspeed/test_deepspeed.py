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
import io
import itertools
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate.accelerator import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import require_cuda, require_deepspeed
from accelerate.test_utils.training import RegressionDataset
from accelerate.utils.dataclasses import DeepSpeedPlugin
from accelerate.utils.deepspeed import (
    DeepSpeedEngineWrapper,
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
    DummyOptim,
    DummyScheduler,
)
from parameterized import parameterized
from transformers import AutoModel, AutoModelForCausalLM, get_scheduler
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.testing_utils import mockenv_context
from transformers.trainer_utils import set_seed
from transformers.utils import is_torch_bf16_available


set_seed(42)

T5_SMALL = "t5-small"
T5_TINY = "patrickvonplaten/t5-tiny-random"
GPT2_TINY = "sshleifer/tiny-gpt2"

ZERO2 = "zero2"
ZERO3 = "zero3"

FP16 = "fp16"
BF16 = "bf16"

CUSTOM_OPTIMIZER = "custom_optimizer"
CUSTOM_SCHEDULER = "custom_scheduler"
DS_OPTIMIZER = "deepspeed_optimizer"
DS_SCHEDULER = "deepspeed_scheduler"

stages = [ZERO2, ZERO3]
optims = [CUSTOM_OPTIMIZER, DS_OPTIMIZER]
schedulers = [CUSTOM_SCHEDULER, DS_SCHEDULER]
if is_torch_bf16_available():
    dtypes = [FP16, BF16]
else:
    dtypes = [FP16]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Cartesian-product of zero stages with models to test
params = list(itertools.product(stages, dtypes))
optim_scheduler_params = list(itertools.product(optims, schedulers))


@require_deepspeed
@require_cuda
class DeepSpeedConfigIntegration(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self.test_file_dir_str = str(path.parents[0])

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        with io.open(self.ds_config_file[ZERO2], "r", encoding="utf-8") as f:
            config_zero2 = json.load(f)
        with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
            config_zero3 = json.load(f)
            # The following setting slows things down, so don't enable it by default unless needed by a test.
            # It's in the file as a demo for users since we want everything to work out of the box even if slower.
            config_zero3["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False

        self.ds_config_dict = dict(zero2=config_zero2, zero3=config_zero3)

        self.dist_env = dict(
            USE_DEEPSPEED="true",
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def get_config_dict(self, stage):
        # As some tests modify the dict, always make a copy
        return deepcopy(self.ds_config_dict[stage])

    @parameterized.expand(stages, name_func=parameterized_custom_name_func)
    def test_deepspeed_plugin(self, stage):

        # Test zero3_init_flag will be set to False when ZeRO stage != 3
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
            zero3_init_flag=True,
        )
        self.assertFalse(deepspeed_plugin.zero3_init_flag)
        deepspeed_plugin.deepspeed_config = None

        # Test zero3_init_flag will be set to True only when ZeRO stage == 3
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
            zero3_init_flag=True,
        )
        self.assertTrue(deepspeed_plugin.zero3_init_flag)
        deepspeed_plugin.deepspeed_config = None

        # Test config files are loaded correctly
        deepspeed_plugin = DeepSpeedPlugin(config_file=self.ds_config_file[stage], zero3_init_flag=True)
        if stage == ZERO2:
            self.assertFalse(deepspeed_plugin.zero3_init_flag)
        elif stage == ZERO3:
            self.assertTrue(deepspeed_plugin.zero3_init_flag)
            deepspeed_plugin.deepspeed_config = None

        # Test `gradient_accumulation_steps` is set to 1 if unavailable in config file
        with tempfile.TemporaryDirectory() as dirpath:
            ds_config = self.get_config_dict(stage)
            del ds_config["gradient_accumulation_steps"]
            with open(os.path.join(dirpath, "ds_config.json"), "w") as out_file:
                json.dump(ds_config, out_file)
            deepspeed_plugin = DeepSpeedPlugin(config_file=os.path.join(dirpath, "ds_config.json"))
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"], 1)
            deepspeed_plugin.deepspeed_config = None

        # Test `ValueError` is raised if `zero_optimization` is unavailable in config file
        with tempfile.TemporaryDirectory() as dirpath:
            ds_config = self.get_config_dict(stage)
            del ds_config["zero_optimization"]
            with open(os.path.join(dirpath, "ds_config.json"), "w") as out_file:
                json.dump(ds_config, out_file)
            with self.assertRaises(ValueError) as cm:
                deepspeed_plugin = DeepSpeedPlugin(config_file=os.path.join(dirpath, "ds_config.json"))
            self.assertTrue(
                "Please specify the ZeRO optimization config in the DeepSpeed config file." in str(cm.exception)
            )
            deepspeed_plugin.deepspeed_config = None

        # Test `deepspeed_config_process`
        deepspeed_plugin = DeepSpeedPlugin(config_file=self.ds_config_file[stage])
        kwargs = {
            "fp16.enabled": True,
            "bf16.enabled": False,
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_clipping": 1.0,
            "train_batch_size": 16,
            "zero_optimization.reduce_bucket_size": 5e5,
            "zero_optimization.stage3_prefetch_bucket_size": 5e5,
            "zero_optimization.stage3_param_persistence_threshold": 5e5,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }
        deepspeed_plugin.deepspeed_config_process(**kwargs)
        for ds_key_long, value in kwargs.items():
            config, ds_key = deepspeed_plugin.find_config_node(ds_key_long)
            if config.get(ds_key) is not None:
                self.assertEqual(config.get(ds_key), value)

        # Test mismatches
        mismatches = {
            "optimizer.params.lr": 1e-5,
            "optimizer.params.weight_decay": 1e-5,
            "gradient_accumulation_steps": 2,
        }
        with self.assertRaises(ValueError) as cm:
            new_kwargs = deepcopy(kwargs)
            new_kwargs.update(mismatches)
            deepspeed_plugin.deepspeed_config_process(**new_kwargs)
        for key in mismatches.keys():
            self.assertTrue(
                key in str(cm.exception),
                f"{key} is not in the exception message:\n{cm.exception}",
            )

        # Test `ValueError` is raised if some config file fields with `auto` value is missing in `kwargs`
        deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = "auto"
        with self.assertRaises(ValueError) as cm:
            del kwargs["gradient_accumulation_steps"]
            deepspeed_plugin.deepspeed_config_process(**kwargs)
        self.assertTrue("`gradient_accumulation_steps` not found in kwargs." in str(cm.exception))

    @parameterized.expand([FP16, BF16], name_func=parameterized_custom_name_func)
    def test_accelerate_state_deepspeed(self, dtype):
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            zero_stage=ZERO2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
            zero3_init_flag=True,
        )
        with mockenv_context(**self.dist_env):
            state = AcceleratorState(mixed_precision=dtype, deepspeed_plugin=deepspeed_plugin, _from_accelerator=True)
            self.assertTrue(state.deepspeed_plugin.deepspeed_config[dtype]["enabled"])
            state.initialized = False

    def test_init_zero3(self):
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
            zero3_init_flag=True,
        )

        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
            self.assertTrue("dschf" in accelerator.__dict__)
            self.assertTrue(type(accelerator.dschf) == HfDeepSpeedConfig)

    @parameterized.expand(optim_scheduler_params, name_func=parameterized_custom_name_func)
    def test_prepare_deepspeed(self, optim_type, scheduler_type):
        # 1. Testing with one of the ZeRO Stages is enough to test the `_prepare_deepspeed` function.
        # Here we test using ZeRO Stage 2 with FP16 enabled.
        from deepspeed.runtime.engine import DeepSpeedEngine

        kwargs = {
            "fp16.enabled": True,
            "bf16.enabled": False,
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_clipping": 1.0,
            "train_batch_size": 16,
            "zero_optimization.reduce_bucket_size": 5e5,
            "zero_optimization.stage3_prefetch_bucket_size": 5e5,
            "zero_optimization.stage3_param_persistence_threshold": 5e5,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        if optim_type == CUSTOM_OPTIMIZER and scheduler_type == CUSTOM_SCHEDULER:
            # Test custom optimizer + custom scheduler
            deepspeed_plugin = DeepSpeedPlugin(
                gradient_accumulation_steps=1,
                gradient_clipping=1.0,
                zero_stage=2,
                offload_optimizer_device="cpu",
                offload_param_device="cpu",
                zero3_save_16bit_model=False,
                zero3_init_flag=False,
            )
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_plugin)
                self.assertEqual(accelerator.state.deepspeed_plugin.config_file, "none")

                train_set = RegressionDataset(length=80)
                eval_set = RegressionDataset(length=20)
                train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
                eval_dataloader = DataLoader(eval_set, batch_size=32, shuffle=False)
                model = AutoModel.from_pretrained(GPT2_TINY)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=1000,
                )
                dummy_optimizer = DummyOptim(params=model.parameters())
                dummy_lr_scheduler = DummyScheduler()

                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, dummy_optimizer, train_dataloader, eval_dataloader, lr_scheduler
                    )
                self.assertTrue(
                    "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                    in str(cm.exception)
                )
                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                    )
                self.assertTrue(
                    "You cannot create a `DummyScheduler` without specifying a scheduler in the config file."
                    in str(cm.exception)
                )

                with self.assertRaises(ValueError) as cm:
                    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
                self.assertTrue(
                    "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using DeepSpeed."
                    in str(cm.exception)
                )

                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
                )
                self.assertTrue(accelerator.deepspeed_config["zero_allow_untested_optimizer"])
                self.assertTrue(accelerator.deepspeed_config["train_batch_size"], 16)
                self.assertEqual(type(model), DeepSpeedEngine)
                self.assertEqual(type(optimizer), DeepSpeedOptimizerWrapper)
                self.assertEqual(type(lr_scheduler), AcceleratedScheduler)
                self.assertEqual(type(accelerator.deepspeed_engine_wrapped), DeepSpeedEngineWrapper)

        elif optim_type == DS_OPTIMIZER and scheduler_type == DS_SCHEDULER:
            # Test DeepSpeed optimizer + DeepSpeed scheduler
            deepspeed_plugin = DeepSpeedPlugin(config_file=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
                train_set = RegressionDataset(length=80)
                eval_set = RegressionDataset(length=20)
                train_dataloader = DataLoader(train_set, batch_size=10, shuffle=True)
                eval_dataloader = DataLoader(eval_set, batch_size=5, shuffle=False)
                model = AutoModel.from_pretrained(GPT2_TINY)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=1000,
                )
                dummy_optimizer = DummyOptim(params=model.parameters())
                dummy_lr_scheduler = DummyScheduler()
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * kwargs["gradient_accumulation_steps"]
                    * accelerator.num_processes
                )
                accelerator.state.deepspeed_plugin.deepspeed_config_process(**kwargs)
                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                    )
                self.assertTrue(
                    "You cannot specify an optimizer in the config file and in the code at the same time"
                    in str(cm.exception)
                )

                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, dummy_optimizer, train_dataloader, eval_dataloader, lr_scheduler
                    )
                self.assertTrue(
                    "You cannot specify a scheduler in the config file and in the code at the same time"
                    in str(cm.exception)
                )

                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, dummy_optimizer, train_dataloader, eval_dataloader, lr_scheduler
                    )
                self.assertTrue(
                    "You cannot specify a scheduler in the config file and in the code at the same time"
                    in str(cm.exception)
                )

                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                )
                self.assertTrue(type(model) == DeepSpeedEngine)
                self.assertTrue(type(optimizer) == DeepSpeedOptimizerWrapper)
                self.assertTrue(type(lr_scheduler) == DeepSpeedSchedulerWrapper)
                self.assertTrue(type(accelerator.deepspeed_engine_wrapped) == DeepSpeedEngineWrapper)

        elif optim_type == CUSTOM_OPTIMIZER and scheduler_type == DS_SCHEDULER:
            # Test custom optimizer + DeepSpeed scheduler
            deepspeed_plugin = DeepSpeedPlugin(config_file=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
                train_set = RegressionDataset(length=80)
                eval_set = RegressionDataset(length=20)
                train_dataloader = DataLoader(train_set, batch_size=10, shuffle=True)
                eval_dataloader = DataLoader(eval_set, batch_size=5, shuffle=False)
                model = AutoModel.from_pretrained(GPT2_TINY)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=1000,
                )
                dummy_optimizer = DummyOptim(params=model.parameters())
                dummy_lr_scheduler = DummyScheduler()
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * kwargs["gradient_accumulation_steps"]
                    * accelerator.num_processes
                )
                accelerator.state.deepspeed_plugin.deepspeed_config_process(**kwargs)
                del accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                )
                self.assertTrue(type(model) == DeepSpeedEngine)
                self.assertTrue(type(optimizer) == DeepSpeedOptimizerWrapper)
                self.assertTrue(type(lr_scheduler) == DeepSpeedSchedulerWrapper)
                self.assertTrue(type(accelerator.deepspeed_engine_wrapped) == DeepSpeedEngineWrapper)
        elif optim_type == DS_OPTIMIZER and scheduler_type == CUSTOM_SCHEDULER:
            # Test deepspeed optimizer + custom scheduler
            deepspeed_plugin = DeepSpeedPlugin(config_file=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
                train_set = RegressionDataset(length=80)
                eval_set = RegressionDataset(length=20)
                train_dataloader = DataLoader(train_set, batch_size=10, shuffle=True)
                eval_dataloader = DataLoader(eval_set, batch_size=5, shuffle=False)
                model = AutoModel.from_pretrained(GPT2_TINY)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
                lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=1000,
                )
                dummy_optimizer = DummyOptim(params=model.parameters())
                dummy_lr_scheduler = DummyScheduler()
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * kwargs["gradient_accumulation_steps"]
                    * accelerator.num_processes
                )
                accelerator.state.deepspeed_plugin.deepspeed_config_process(**kwargs)
                del accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]
                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, dummy_optimizer, train_dataloader, eval_dataloader, lr_scheduler
                    )
                self.assertTrue(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using `accelerate.utils.DummyOptim`."
                    in str(cm.exception)
                )
        accelerator.state.initialized = False

    def test_save_checkpoints(self):
        deepspeed_plugin = DeepSpeedPlugin(
            config_file=self.ds_config_file[ZERO3],
            zero3_init_flag=True,
        )
        del deepspeed_plugin.deepspeed_config["bf16"]
        kwargs = {
            "fp16.enabled": True,
            "bf16.enabled": False,
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_clipping": 1.0,
            "train_batch_size": 16,
            "zero_optimization.reduce_bucket_size": 5e5,
            "zero_optimization.stage3_prefetch_bucket_size": 5e5,
            "zero_optimization.stage3_param_persistence_threshold": 5e5,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
            kwargs["train_batch_size"] = (
                kwargs["train_micro_batch_size_per_gpu"]
                * kwargs["gradient_accumulation_steps"]
                * accelerator.num_processes
            )
            accelerator.state.deepspeed_plugin.deepspeed_config_process(**kwargs)

            train_set = RegressionDataset(length=80)
            eval_set = RegressionDataset(length=20)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=32, shuffle=False)
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            dummy_optimizer = DummyOptim(params=model.parameters())
            dummy_lr_scheduler = DummyScheduler()

            model, _, train_dataloader, eval_dataloader, _ = accelerator.prepare(
                model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
            )
            with self.assertRaises(ValueError) as cm:
                accelerator.get_state_dict(model)
            msg = (
                "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
            )
            self.assertTrue(msg in str(cm.exception))
