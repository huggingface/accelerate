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

import os
import unittest

from parameterized import parameterized

from accelerate.test_utils.testing import require_huggingface_suite, require_multi_gpu, slow

from .test_distributed import CONFIGS_DIR, SCRIPTS_DIR, DistributedTrainingCommon, run_script


FSDP2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp2.yaml")


@slow
@require_multi_gpu
@require_huggingface_suite
class FSDP2TrainingTest(DistributedTrainingCommon):
    @parameterized.expand(["fp32", "bf16", "fp16"])
    def test_training(self, dtype):
        self.check_training(dtype, config_file=FSDP2_CONFIG_FILE)

    @parameterized.expand(["fp16", "bf16"])
    def test_training_mixed_precision(self, mixed_precision):
        self.check_mixed_precision(mixed_precision, config_file=FSDP2_CONFIG_FILE)

    def test_training_with_gradient_accumulation(self):
        self.check_gradient_accumulation(config_file=FSDP2_CONFIG_FILE)

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=FSDP2_CONFIG_FILE)


@slow
@require_multi_gpu
@require_huggingface_suite
class FSDP2LoadingTest(unittest.TestCase):
    cpu_ram_efficient_loading = False

    def check_loading(self, **script_args):
        run_script(
            os.path.join(SCRIPTS_DIR, "load.py"),
            config_file=FSDP2_CONFIG_FILE,
            cpu_ram_efficient_loading=str(self.cpu_ram_efficient_loading).lower(),
            timeout=180,
            **script_args,
        )

    def test_loading(self):
        self.check_loading()

    def test_loading_with_cpu_offload(self):
        self.check_loading(cpu_offload="true")

    # Ignored parameters must stay on the accelerator even with CPU offload: FSDP does not move them back for forward.
    @parameterized.expand(["false", "true"])
    def test_loading_with_ignored_modules(self, cpu_offload):
        self.check_loading(ignored_modules="true", cpu_offload=cpu_offload)


class FSDP2RamEfficientLoadingTest(FSDP2LoadingTest):
    cpu_ram_efficient_loading = True
