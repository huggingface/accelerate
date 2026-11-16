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

import torch
from parameterized import parameterized

from accelerate import Accelerator
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    get_torch_dist_unique_port,
    require_huggingface_suite,
    require_multi_gpu,
    require_non_cpu,
    require_non_torch_xla,
    slow,
)
from accelerate.utils import patch_environment

from .test_distributed import CONFIGS_DIR, DistributedTrainingCommon


FSDP2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp2.yaml")


@require_non_cpu
@require_non_torch_xla
class FSDP2UnitTest(AccelerateTestCase):
    def test_activation_checkpointing_shards_the_wrapper(self):
        """A checkpointed layer is sharded at its wrapper, unless it is also called without the wrapper."""
        from torch.distributed.fsdp import FSDPModule

        shared = torch.nn.Linear(4, 4)
        # Checkpointing wraps `model[1]` only, so calls through `model[2]` bypass the wrapper's FSDP hooks.
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), shared, shared)
        env = {
            "ACCELERATE_USE_FSDP": "true",
            "FSDP_VERSION": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(get_torch_dist_unique_port()),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "FSDP_AUTO_WRAP_POLICY": "TRANSFORMER_BASED_WRAP",
            "FSDP_TRANSFORMER_CLS_TO_WRAP": "Linear",
            "FSDP_ACTIVATION_CHECKPOINTING": "true",
        }
        with patch_environment(**env):
            owns_process_group = not torch.distributed.is_initialized()
            accelerator = Accelerator()
            if owns_process_group:
                self.addCleanup(torch.distributed.destroy_process_group)
            model, _ = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))

        assert isinstance(model[0], FSDPModule) and not isinstance(model[0]._checkpoint_wrapped_module, FSDPModule)
        assert isinstance(model[2], FSDPModule)


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
