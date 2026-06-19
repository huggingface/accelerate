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

import unittest

from parameterized import parameterized

from accelerate.test_utils.testing import (
    TempDirTestCase,
    device_count,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
    require_multi_device,
    require_non_torch_xla,
    require_transformers,
    run_first,
)
from accelerate.utils import patch_environment


@require_non_torch_xla
@require_multi_device
@require_transformers
@run_first
class AccelerateUlyssesSPTest(TempDirTestCase):
    """Native ("accelerate") Ulysses SP under DDP / FSDP2 (DeepSpeed lives in tests/deepspeed). Each
    run also asserts the auto-wrapped `SequenceShardingDataLoader` behaves (same sample per sp rank,
    sequence sharded, shift_labels/position_ids built)."""

    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    def _launch(self, num_processes, script_args):
        script = self.test_scripts_folder / "test_accelerate_ulysses_sp.py"
        cmd = get_launch_command(num_processes=num_processes, num_machines=1, machine_rank=0, mixed_precision="bf16")
        cmd.extend([str(script), *script_args])
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)

    @parameterized.expand(["ddp", "fsdp"])
    def test_ulysses_sp(self, engine):  # 2 GPUs: pure sp=2
        self._launch(2, [f"--engine={engine}"])

    def test_ulysses_sp_packed(self):  # 2 GPUs: packed / varlen sequences
        self._launch(2, ["--engine=fsdp", "--packed"])

    @unittest.skipUnless(device_count >= 4, "dp x sp requires >= 4 devices")
    @parameterized.expand(["ddp", "fsdp"])
    def test_ulysses_sp_x_dp(self, engine):  # 4 GPUs: dp=2 x sp=2
        self._launch(4, [f"--engine={engine}", "--sp-size=2"])
