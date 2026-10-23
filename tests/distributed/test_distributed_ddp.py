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

from parameterized import parameterized

from accelerate.test_utils.testing import require_huggingface_suite, require_multi_gpu, slow

from .test_distributed import CONFIGS_DIR, DistributedTrainingCommon


DDP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "ddp.yaml")


@slow
@require_multi_gpu
@require_huggingface_suite
class DDPTrainingTest(DistributedTrainingCommon):
    @parameterized.expand(["fp32", "bf16", "fp16"])
    def test_training(self, dtype):
        self.check_training(dtype, config_file=DDP_CONFIG_FILE)

    @parameterized.expand(["fp16", "bf16"])
    def test_training_mixed_precision(self, mixed_precision):
        self.check_mixed_precision(mixed_precision, config_file=DDP_CONFIG_FILE)

    def test_training_with_gradient_accumulation(self):
        self.check_gradient_accumulation(config_file=DDP_CONFIG_FILE)

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=DDP_CONFIG_FILE)
