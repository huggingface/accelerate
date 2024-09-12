# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import json
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    execute_subprocess_async,
    path_in_accelerate_package,
    require_deepspeed,
    require_huggingface_suite,
    require_multi_device,
    require_non_cpu,
    slow,
)
from accelerate.test_utils.training import RegressionDataset
from accelerate.utils import patch_environment
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler, get_active_deepspeed_plugin


GPT2_TINY = "hf-internal-testing/tiny-random-gpt2"


@require_deepspeed
@require_non_cpu
class DeepSpeedConfigIntegration(AccelerateTestCase):
    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    def setUp(self):
        super().setUp()

        self.dist_env = dict(
            ACCELERATE_USE_DEEPSPEED="true",
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self.test_file_dir_str = str(path.parents[0])

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3_inference=f"{self.test_file_dir_str}/ds_config_zero3_model_only.json",
            zero3_training=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        with open(self.ds_config_file["zero2"], encoding="utf-8") as f:
            self.config_zero2 = json.load(f)
        with open(self.ds_config_file["zero3_training"], encoding="utf-8") as f:
            self.config_zero3 = json.load(f)
        with open(self.ds_config_file["zero3_inference"], encoding="utf-8") as f:
            self.config_zero3_inference = json.load(f)

        self.model_init = partial(AutoModelForCausalLM.from_pretrained, GPT2_TINY)

    def get_ds_plugins(self, zero3_inference=False):
        ds_zero2 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero2,
        )
        ds_zero3 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero3 if not zero3_inference else self.config_zero3_inference,
        )
        return {"zero2": ds_zero2, "zero3": ds_zero3}

    def test_enable_disable(self):
        ds_plugins = self.get_ds_plugins()
        ds_zero2, ds_zero3 = ds_plugins.values()
        accelerator = Accelerator(
            deepspeed_plugin=ds_plugins,
        )
        # Accelerator's constructor should automatically enable the first plugin
        assert ds_zero2.enabled
        assert not ds_zero3.enabled
        assert get_active_deepspeed_plugin(accelerator.state) == ds_zero2
        assert accelerator.deepspeed_plugin == ds_zero2
        assert accelerator.state.get_deepspeed_plugin("zero2") == ds_zero2
        accelerator.state.enable_deepspeed_plugin("zero3")
        assert not ds_zero2.enabled
        assert ds_zero3.enabled
        assert get_active_deepspeed_plugin(accelerator.state) == ds_zero3
        assert accelerator.deepspeed_plugin == ds_zero3
        assert accelerator.state.get_deepspeed_plugin("zero3") == ds_zero3
        accelerator.state.enable_deepspeed_plugin("zero2")
        assert not ds_zero3.enabled
        assert ds_zero2.enabled
        assert get_active_deepspeed_plugin(accelerator.state) == ds_zero2
        assert accelerator.deepspeed_plugin == ds_zero2
        assert accelerator.state.get_deepspeed_plugin("zero2") == ds_zero2

    def test_enable_disable_manually_set(self):
        ds_plugins = self.get_ds_plugins()
        ds_zero2, _ = ds_plugins.values()
        with self.assertRaises(ValueError):
            ds_zero2.enable()
        accelerator = Accelerator(deepspeed_plugin=ds_plugins)
        accelerator.state.enable_deepspeed_plugin("zero2")
        with self.assertRaises(NotImplementedError):
            ds_zero2.enabled = False
        assert ds_zero2.enabled

    def test_prepare_multiple_models_zero3_inference(self):
        ds_plugins = self.get_ds_plugins(zero3_inference=True)
        accelerator = Accelerator(deepspeed_plugin=ds_plugins)
        # Using Zero-2 first
        model1 = self.model_init()
        optimizer = DummyOptim(model1.parameters())
        scheduler = DummyScheduler(optimizer)

        dataset = RegressionDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        model1, optimizer, scheduler, dataloader = accelerator.prepare(model1, optimizer, scheduler, dataloader)
        accelerator.state.enable_deepspeed_plugin("zero3")
        model2 = self.model_init()
        with self.assertLogs(level="WARNING") as captured:
            model2 = accelerator.prepare(model2)
            self.assertIn(
                "A wrapped DeepSpeed engine reference is currently tied for this `Accelerator()` instance.",
                captured.output[0],
            )

        assert accelerator.deepspeed_engine_wrapped.engine is model1

    @require_huggingface_suite
    @require_multi_device
    @slow
    def test_train_multiple_models(self):
        self.test_file_path = self.test_scripts_folder / "test_ds_multiple_model.py"
        cmd = ["accelerate", "launch", "--num_processes=2", "--num_machines=1", self.test_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)
