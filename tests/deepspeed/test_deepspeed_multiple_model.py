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
import json
from functools import partial
from pathlib import Path

from transformers import AutoModelForCausalLM

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.test_utils.testing import AccelerateTestCase, require_deepspeed, require_non_cpu
from accelerate.utils.deepspeed import get_active_deepspeed_plugin


GPT2_TINY = "sshleifer/tiny-gpt2"


@require_deepspeed
@require_non_cpu
class DeepSpeedConfigIntegration(AccelerateTestCase):
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
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        with open(self.ds_config_file["zero2"], encoding="utf-8") as f:
            self.config_zero2 = json.load(f)
        with open(self.ds_config_file["zero3"], encoding="utf-8") as f:
            self.config_zero3 = json.load(f)

        self.model_init = partial(AutoModelForCausalLM.from_pretrained, GPT2_TINY)

    def test_enable_disable(self):
        ds_zero2 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero2,
        )
        ds_zero3 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero3,
        )
        _ = Accelerator(
            deepspeed_plugin=[ds_zero2, ds_zero3],
        )
        # Accelerator should flag the first plugin as enabled
        assert ds_zero2.enabled
        assert not ds_zero3.enabled
        ds_zero3.enable()
        assert not ds_zero2.enabled
        assert ds_zero3.enabled
        ds_zero2.enable()
        assert not ds_zero3.enabled
        assert ds_zero2.enabled

    def test_get_active_plugin(self):
        ds_zero2 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero2,
        )
        ds_zero3 = DeepSpeedPlugin(
            hf_ds_config=self.config_zero3,
        )
        accelerator = Accelerator(deepspeed_plugin=[ds_zero2, ds_zero3])
        assert get_active_deepspeed_plugin(accelerator.state) == ds_zero2
        ds_zero3.enable()
        assert get_active_deepspeed_plugin(accelerator.state) == ds_zero3
