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

from parameterized import param, parameterized

from accelerate.test_utils.testing import require_deepspeed, require_huggingface_suite, require_multi_gpu, slow

from .test_distributed import CONFIGS_DIR, DistributedTrainingCommon


ZERO2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "zero2.yaml")
ZERO3_CONFIG_FILE = os.path.join(CONFIGS_DIR, "zero3.yaml")

# Unlike the reference, DeepSpeed runs every op in bf16/fp16 instead of using autocast, and updates fp32 master weights.
# The losses drift up to ~2e-3 after 8 steps, while a bug moves them by more than 5e-2.
LOW_PRECISION_ATOL = 1e-2


@require_deepspeed
@slow
@require_multi_gpu
@require_huggingface_suite
class DeepSpeedZeRO2TrainingTest(DistributedTrainingCommon):
    @parameterized.expand(["fp32", param("bf16", atol=LOW_PRECISION_ATOL), param("fp16", atol=LOW_PRECISION_ATOL)])
    def test_training(self, dtype, **kwargs):
        self.check_training(dtype, config_file=ZERO2_CONFIG_FILE, **kwargs)

    # The reference also uses mixed precision, but with autocast: DeepSpeed instead casts the whole model to 16-bit.
    @parameterized.expand([param("fp16", atol=LOW_PRECISION_ATOL), param("bf16", atol=LOW_PRECISION_ATOL)])
    def test_training_mixed_precision(self, mixed_precision, **kwargs):
        self.check_mixed_precision(mixed_precision, config_file=ZERO2_CONFIG_FILE, **kwargs)

    def test_training_with_gradient_accumulation(self):
        # In fp32, so DeepSpeed's low precision drift can't hide an accumulation bug.
        self.check_gradient_accumulation(config_file=ZERO2_CONFIG_FILE, dtype="fp32")

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=ZERO2_CONFIG_FILE)


@require_deepspeed
@slow
@require_multi_gpu
@require_huggingface_suite
class DeepSpeedZeRO3TrainingTest(DistributedTrainingCommon):
    @parameterized.expand(["fp32", param("bf16", atol=LOW_PRECISION_ATOL), param("fp16", atol=LOW_PRECISION_ATOL)])
    def test_training(self, dtype, **kwargs):
        self.check_training(dtype, config_file=ZERO3_CONFIG_FILE, **kwargs)

    # The reference also uses mixed precision, but with autocast: DeepSpeed instead casts the whole model to 16-bit.
    @parameterized.expand([param("fp16", atol=LOW_PRECISION_ATOL), param("bf16", atol=LOW_PRECISION_ATOL)])
    def test_training_mixed_precision(self, mixed_precision, **kwargs):
        self.check_mixed_precision(mixed_precision, config_file=ZERO3_CONFIG_FILE, **kwargs)

    def test_training_with_gradient_accumulation(self):
        # In fp32, so DeepSpeed's low precision drift can't hide an accumulation bug.
        self.check_gradient_accumulation(config_file=ZERO3_CONFIG_FILE, dtype="fp32")

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=ZERO3_CONFIG_FILE)
