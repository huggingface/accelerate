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
from copy import deepcopy
from pathlib import Path

import torch
from parameterized import parameterized
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, get_scheduler
from transformers.testing_utils import mockenv_context
from transformers.trainer_utils import set_seed
from transformers.utils import is_torch_bf16_available

import accelerate
from accelerate.accelerator import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    TempDirTestCase,
    execute_subprocess_async,
    require_deepspeed,
    require_multi_device,
    require_non_cpu,
    slow,
)
from accelerate.test_utils.training import RegressionDataset
from accelerate.utils.dataclasses import DeepSpeedPlugin
from accelerate.utils.deepspeed import (
    DeepSpeedEngineWrapper,
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
    DummyOptim,
    DummyScheduler,
)
from accelerate.utils.other import patch_environment


set_seed(42)

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
@require_non_cpu
class DeepSpeedConfigIntegration(AccelerateTestCase):
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
            ACCELERATE_USE_DEEPSPEED="true",
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
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_file[stage], zero3_init_flag=True)
        if stage == ZERO2:
            self.assertFalse(deepspeed_plugin.zero3_init_flag)
        elif stage == ZERO3:
            self.assertTrue(deepspeed_plugin.zero3_init_flag)

        # Test `gradient_accumulation_steps` is set to 1 if unavailable in config file
        with tempfile.TemporaryDirectory() as dirpath:
            ds_config = self.get_config_dict(stage)
            del ds_config["gradient_accumulation_steps"]
            with open(os.path.join(dirpath, "ds_config.json"), "w") as out_file:
                json.dump(ds_config, out_file)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=os.path.join(dirpath, "ds_config.json"))
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"], 1)
            deepspeed_plugin.deepspeed_config = None

        # Test `ValueError` is raised if `zero_optimization` is unavailable in config file
        with tempfile.TemporaryDirectory() as dirpath:
            ds_config = self.get_config_dict(stage)
            del ds_config["zero_optimization"]
            with open(os.path.join(dirpath, "ds_config.json"), "w") as out_file:
                json.dump(ds_config, out_file)
            with self.assertRaises(ValueError) as cm:
                deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=os.path.join(dirpath, "ds_config.json"))
            self.assertTrue(
                "Please specify the ZeRO optimization config in the DeepSpeed config." in str(cm.exception)
            )
            deepspeed_plugin.deepspeed_config = None

        # Test `deepspeed_config_process`
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_file[stage])
        kwargs = {
            "fp16.enabled": True,
            "bf16.enabled": False,
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
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
            config, ds_key = deepspeed_plugin.hf_ds_config.find_config_node(ds_key_long)
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
        deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"] = "auto"
        with self.assertRaises(ValueError) as cm:
            del kwargs["optimizer.params.lr"]
            deepspeed_plugin.deepspeed_config_process(**kwargs)
        self.assertTrue("`optimizer.params.lr` not found in kwargs." in str(cm.exception))

    @parameterized.expand([FP16, BF16], name_func=parameterized_custom_name_func)
    def test_accelerate_state_deepspeed(self, dtype):
        AcceleratorState._reset_state(True)
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
            state = Accelerator(mixed_precision=dtype, deepspeed_plugin=deepspeed_plugin).state
            self.assertTrue(state.deepspeed_plugin.deepspeed_config[dtype]["enabled"])

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
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)  # noqa: F841
            from transformers.deepspeed import is_deepspeed_zero3_enabled

            self.assertTrue(is_deepspeed_zero3_enabled())

    @parameterized.expand(optim_scheduler_params, name_func=parameterized_custom_name_func)
    def test_prepare_deepspeed(self, optim_type, scheduler_type):
        # 1. Testing with one of the ZeRO Stages is enough to test the `_prepare_deepspeed` function.
        # Here we test using ZeRO Stage 2 with FP16 enabled.
        from deepspeed.runtime.engine import DeepSpeedEngine

        kwargs = {
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
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
                dummy_lr_scheduler = DummyScheduler(dummy_optimizer)

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
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                    in str(cm.exception)
                )

                with self.assertRaises(ValueError) as cm:
                    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
                self.assertTrue(
                    "When using DeepSpeed `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                    in str(cm.exception)
                )

                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
                )
                self.assertTrue(accelerator.deepspeed_config["zero_allow_untested_optimizer"])
                self.assertTrue(accelerator.deepspeed_config["train_batch_size"], 16)
                self.assertEqual(type(model), DeepSpeedEngine)
                self.assertEqual(type(optimizer), DeepSpeedOptimizerWrapper)
                self.assertEqual(type(lr_scheduler), DeepSpeedSchedulerWrapper)
                self.assertEqual(type(accelerator.deepspeed_engine_wrapped), DeepSpeedEngineWrapper)

        elif optim_type == DS_OPTIMIZER and scheduler_type == DS_SCHEDULER:
            # Test DeepSpeed optimizer + DeepSpeed scheduler
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")
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
                dummy_lr_scheduler = DummyScheduler(dummy_optimizer)
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
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
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")
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
                dummy_lr_scheduler = DummyScheduler(dummy_optimizer)
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
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
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.ds_config_file[ZERO2])
            with mockenv_context(**self.dist_env):
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")
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
                dummy_lr_scheduler = DummyScheduler(dummy_optimizer)
                kwargs["train_batch_size"] = (
                    kwargs["train_micro_batch_size_per_gpu"]
                    * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
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

                # passing `DummyScheduler` without `lr_scheduler_callable` should fail
                with self.assertRaises(ValueError) as cm:
                    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                        model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                    )
                self.assertTrue(
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                    in str(cm.exception)
                )

                # passing `lr_scheduler_callable` to DummyScheduler should enable DS Optim + Custom Scheduler
                def _lr_scheduler_callable(optimizer):
                    return get_scheduler(
                        name="linear",
                        optimizer=optimizer,
                        num_warmup_steps=0,
                        num_training_steps=1000,
                    )

                dummy_lr_scheduler = DummyScheduler(dummy_optimizer, lr_scheduler_callable=_lr_scheduler_callable)
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                    model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
                )

    def test_save_checkpoints(self):
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=self.ds_config_file[ZERO3],
            zero3_init_flag=True,
        )
        del deepspeed_plugin.deepspeed_config["bf16"]
        kwargs = {
            "optimizer.params.lr": 5e-5,
            "optimizer.params.weight_decay": 0.0,
            "scheduler.params.warmup_min_lr": 0.0,
            "scheduler.params.warmup_max_lr": 5e-5,
            "scheduler.params.warmup_num_steps": 0,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_clipping": 1.0,
            "train_batch_size": 16,
            "zero_optimization.reduce_bucket_size": 5e5,
            "zero_optimization.stage3_prefetch_bucket_size": 5e5,
            "zero_optimization.stage3_param_persistence_threshold": 5e5,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision="fp16")
            kwargs["train_batch_size"] = (
                kwargs["train_micro_batch_size_per_gpu"]
                * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
                * accelerator.num_processes
            )
            accelerator.state.deepspeed_plugin.deepspeed_config_process(**kwargs)

            train_set = RegressionDataset(length=80)
            eval_set = RegressionDataset(length=20)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=32, shuffle=False)
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            dummy_optimizer = DummyOptim(params=model.parameters())
            dummy_lr_scheduler = DummyScheduler(dummy_optimizer)

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

    def test_autofill_dsconfig(self):
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=self.ds_config_file[ZERO3],
            zero3_init_flag=True,
        )
        del deepspeed_plugin.deepspeed_config["bf16"]
        del deepspeed_plugin.deepspeed_config["fp16"]

        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
            train_set = RegressionDataset(length=80)
            eval_set = RegressionDataset(length=20)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=32, shuffle=False)
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            dummy_optimizer = DummyOptim(params=model.parameters(), lr=5e-5, weight_decay=1e-4)
            dummy_lr_scheduler = DummyScheduler(dummy_optimizer, warmup_num_steps=10, total_num_steps=1000)
            hidden_size = model.config.hidden_size
            model, _, train_dataloader, eval_dataloader, _ = accelerator.prepare(
                model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
            )
            self.assertEqual(accelerator.deepspeed_config["train_micro_batch_size_per_gpu"], 16)
            self.assertEqual(accelerator.deepspeed_config["train_batch_size"], 16)

            self.assertEqual(accelerator.deepspeed_config["optimizer"]["params"]["lr"], 5e-5)
            self.assertEqual(accelerator.deepspeed_config["optimizer"]["params"]["weight_decay"], 1e-4)

            self.assertEqual(accelerator.deepspeed_config["scheduler"]["params"]["warmup_min_lr"], 0.0)
            self.assertEqual(accelerator.deepspeed_config["scheduler"]["params"]["warmup_max_lr"], 5e-5)
            self.assertEqual(accelerator.deepspeed_config["scheduler"]["params"]["warmup_num_steps"], 10)

            self.assertEqual(accelerator.deepspeed_config["gradient_clipping"], 1.0)
            self.assertEqual(
                accelerator.deepspeed_config["zero_optimization"]["reduce_bucket_size"], hidden_size * hidden_size
            )
            self.assertEqual(
                accelerator.deepspeed_config["zero_optimization"]["stage3_prefetch_bucket_size"],
                0.9 * hidden_size * hidden_size,
            )
            self.assertEqual(
                accelerator.deepspeed_config["zero_optimization"]["stage3_param_persistence_threshold"],
                10 * hidden_size,
            )
            self.assertFalse(
                accelerator.deepspeed_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"]
            )

    @parameterized.expand([FP16, BF16], name_func=parameterized_custom_name_func)
    def test_autofill_dsconfig_from_ds_plugin(self, dtype):
        ds_config = self.ds_config_dict["zero3"]
        if dtype == BF16:
            del ds_config["fp16"]
        else:
            del ds_config["bf16"]
        ds_config[dtype]["enabled"] = "auto"
        ds_config["zero_optimization"]["stage"] = "auto"
        ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = "auto"
        ds_config["zero_optimization"]["offload_optimizer"]["device"] = "auto"
        ds_config["zero_optimization"]["offload_param"]["device"] = "auto"
        ds_config["gradient_accumulation_steps"] = "auto"
        ds_config["gradient_clipping"] = "auto"

        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config,
            zero3_init_flag=True,
            gradient_accumulation_steps=2,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
        )

        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision=dtype)
            deepspeed_plugin = accelerator.state.deepspeed_plugin
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_clipping"], 1.0)
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"], 2)
            self.assertEqual(deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"], 2)
            self.assertEqual(
                deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_optimizer"]["device"], "cpu"
            )
            self.assertEqual(deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["device"], "cpu")
            self.assertTrue(
                deepspeed_plugin.deepspeed_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"]
            )
            self.assertTrue(deepspeed_plugin.deepspeed_config[dtype]["enabled"])

        AcceleratorState._reset_state(True)
        diff_dtype = "bf16" if dtype == "fp16" else "fp16"
        with mockenv_context(**self.dist_env):
            with self.assertRaises(ValueError) as cm:
                accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision=diff_dtype)
            self.assertTrue(
                f"`--mixed_precision` arg cannot be set to `{diff_dtype}` when `{dtype}` is set in the DeepSpeed config file."
                in str(cm.exception)
            )

        # base case of passing in `gradient_accumulation_steps` to `DeepSpeedPlugin`
        AcceleratorState._reset_state(True)
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=4)
        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision=dtype)
            deepspeed_plugin = accelerator.state.deepspeed_plugin
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"], 4)

        # filling the `auto` gradient_accumulation_steps via Accelerator's value
        AcceleratorState._reset_state(True)
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config,
            zero3_init_flag=True,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=True,
        )
        with mockenv_context(**self.dist_env):
            accelerator = Accelerator(
                deepspeed_plugin=deepspeed_plugin, mixed_precision=dtype, gradient_accumulation_steps=8
            )
            train_set = RegressionDataset(length=80)
            eval_set = RegressionDataset(length=20)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
            eval_dataloader = DataLoader(eval_set, batch_size=32, shuffle=False)
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            dummy_optimizer = DummyOptim(params=model.parameters(), lr=5e-5, weight_decay=1e-4)
            dummy_lr_scheduler = DummyScheduler(dummy_optimizer, warmup_num_steps=10, total_num_steps=1000)
            model, _, train_dataloader, eval_dataloader, _ = accelerator.prepare(
                model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
            )
            deepspeed_plugin = accelerator.state.deepspeed_plugin
            self.assertEqual(deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"], 8)

    def test_ds_config_assertions(self):
        ambiguous_env = self.dist_env.copy()
        ambiguous_env[
            "ACCELERATE_CONFIG_DS_FIELDS"
        ] = "gradient_accumulation_steps,gradient_clipping,zero_stage,offload_optimizer_device,offload_param_device,zero3_save_16bit_model,mixed_precision"

        with mockenv_context(**ambiguous_env):
            with self.assertRaises(ValueError) as cm:
                deepspeed_plugin = DeepSpeedPlugin(
                    hf_ds_config=self.ds_config_file[ZERO3],
                    zero3_init_flag=True,
                    gradient_accumulation_steps=1,
                    gradient_clipping=1.0,
                    zero_stage=ZERO2,
                    offload_optimizer_device="cpu",
                    offload_param_device="cpu",
                    zero3_save_16bit_model=True,
                )
                _ = Accelerator(deepspeed_plugin=deepspeed_plugin, mixed_precision=FP16)
            self.assertTrue(
                "If you are using an accelerate config file, remove others config variables mentioned in the above specified list."
                in str(cm.exception)
            )

    @parameterized.expand(stages, name_func=parameterized_custom_name_func)
    def test_ds_config(self, stage):
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=self.ds_config_file[stage],
            zero3_init_flag=True,
        )
        self.assertEqual(deepspeed_plugin.zero_stage, int(stage.replace("zero", "")))

    def test_basic_run(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        test_file_path = os.path.sep.join(
            mod_file.split(os.path.sep)[:-1] + ["scripts", "external_deps", "test_performance.py"]
        )
        with tempfile.TemporaryDirectory() as dirpath:
            cmd = [
                "accelerate",
                "launch",
                "--num_processes=1",
                "--num_machines=1",
                "--machine_rank=0",
                "--mixed_precision=fp16",
                "--use_deepspeed",
                "--gradient_accumulation_steps=1",
                "--zero_stage=2",
                "--offload_optimizer_device=none",
                "--offload_param_device=none",
                test_file_path,
                "--model_name_or_path=distilbert-base-uncased",
                "--num_epochs=1",
                f"--output_dir={dirpath}",
            ]
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd, env=os.environ.copy())


@require_deepspeed
@require_multi_device
@slow
class DeepSpeedIntegrationTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self.test_file_dir_str = str(path.parents[0])

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        self.stages = [1, 2, 3]
        self.zero3_offload_config = False
        self.performance_lower_bound = 0.82
        self.peak_memory_usage_upper_bound = {
            "multi_gpu_fp16": 3200,
            "deepspeed_stage_1_fp16": 1600,
            "deepspeed_stage_2_fp16": 2500,
            "deepspeed_stage_3_zero_init_fp16": 2800,
            # Disabling below test as it overwhelms the RAM memory usage
            # on CI self-hosted runner leading to tests getting killed.
            # "deepspeed_stage_3_cpu_offload_fp16": 1900,
        }
        self.n_train = 160
        self.n_val = 160

        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_scripts_folder = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "external_deps"])

    def test_performance(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_performance.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--mixed_precision=fp16",
            "--use_deepspeed",
            "--gradient_accumulation_steps=1",
            "--gradient_clipping=1",
            "--zero3_init_flag=True",
            "--zero3_save_16bit_model=True",
        ]
        for stage in self.stages:
            if stage == 1:
                continue
            cmd_stage = cmd.copy()
            cmd_stage.extend([f"--zero_stage={stage}"])
            cmd_stage.extend(["--offload_optimizer_device=none", "--offload_param_device=none"])
            if self.zero3_offload_config:
                with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
                    ds_config = json.load(f)
                    del ds_config["bf16"]
                    del ds_config["optimizer"]["params"]["torch_adam"]
                    del ds_config["optimizer"]["params"]["adam_w_mode"]
                    ds_config["fp16"]["enabled"] = True
                    ds_config_path = os.path.join(self.tmpdir, "ds_config.json")
                    with open(ds_config_path, "w") as out_file:
                        json.dump(ds_config, out_file)

                cmd_stage.extend([f"--deepspeed_config_file={ds_config_path}"])

            cmd_stage.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    f"--performance_lower_bound={self.performance_lower_bound}",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_stage, env=os.environ.copy())

    def test_checkpointing(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_checkpointing.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--mixed_precision=fp16",
            "--use_deepspeed",
            "--gradient_accumulation_steps=1",
            "--gradient_clipping=1",
            "--zero3_init_flag=True",
            "--zero3_save_16bit_model=True",
        ]
        for stage in self.stages:
            if stage == 1:
                continue
            cmd_stage = cmd.copy()
            cmd_stage.extend([f"--zero_stage={stage}"])
            cmd_stage.extend(["--offload_optimizer_device=none", "--offload_param_device=none"])
            if self.zero3_offload_config:
                with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
                    ds_config = json.load(f)
                    del ds_config["bf16"]
                    del ds_config["optimizer"]["params"]["torch_adam"]
                    del ds_config["optimizer"]["params"]["adam_w_mode"]
                    ds_config["fp16"]["enabled"] = True
                    ds_config_path = os.path.join(self.tmpdir, "ds_config.json")
                    with open(ds_config_path, "w") as out_file:
                        json.dump(ds_config, out_file)

                cmd_stage.extend([f"--deepspeed_config_file={ds_config_path}"])

            cmd_stage.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    "--partial_train_epoch=1",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_stage, env=os.environ.copy())

            cmd_stage = cmd_stage[:-1]
            resume_from_checkpoint = os.path.join(self.tmpdir, "epoch_0")
            cmd_stage.extend(
                [
                    f"--resume_from_checkpoint={resume_from_checkpoint}",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_stage, env=os.environ.copy())

    def test_peak_memory_usage(self):
        self.test_file_path = os.path.join(self.test_scripts_folder, "test_peak_memory_usage.py")
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
        ]
        for spec, peak_mem_upper_bound in self.peak_memory_usage_upper_bound.items():
            cmd_stage = cmd.copy()
            if "fp16" in spec:
                cmd_stage.extend(["--mixed_precision=fp16"])

            if "multi_gpu" in spec:
                continue
            else:
                cmd_stage.extend(
                    [
                        "--use_deepspeed",
                        "--gradient_accumulation_steps=1",
                        "--gradient_clipping=1",
                        "--zero3_init_flag=True",
                        "--zero3_save_16bit_model=True",
                    ]
                )
                for i in range(3):
                    if f"stage_{i+1}" in spec:
                        cmd_stage.extend([f"--zero_stage={i+1}"])
                        break
                cmd_stage.extend(
                    [
                        "--offload_optimizer_device=none",
                        "--offload_param_device=none",
                        "--offload_optimizer_nvme_path=none",
                        "--offload_param_nvme_path=none",
                    ]
                )
                if "cpu_offload" in spec:
                    with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
                        ds_config = json.load(f)
                        del ds_config["bf16"]
                        del ds_config["fp16"]
                        del ds_config["optimizer"]["params"]["torch_adam"]
                        del ds_config["optimizer"]["params"]["adam_w_mode"]
                        ds_config_path = os.path.join(self.tmpdir, "ds_config.json")
                        with open(ds_config_path, "w") as out_file:
                            json.dump(ds_config, out_file)

                    cmd_stage.extend([f"--deepspeed_config_file={ds_config_path}"])

            cmd_stage.extend(
                [
                    self.test_file_path,
                    f"--output_dir={self.tmpdir}",
                    f"--peak_memory_upper_bound={peak_mem_upper_bound}",
                    f"--n_train={self.n_train}",
                    f"--n_val={self.n_val}",
                ]
            )
            with patch_environment(omp_num_threads=1):
                execute_subprocess_async(cmd_stage, env=os.environ.copy())
