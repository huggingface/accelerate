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

from parameterized import parameterized

from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    path_in_accelerate_package,
    require_deepspeed,
    require_multi_device,
    require_transformers,
)
from accelerate.utils import patch_environment


@require_deepspeed
@require_multi_device
@require_transformers
class AccelerateUlyssesSPDeepSpeedTest(TempDirTestCase):
    """Native ("accelerate") Ulysses SP running under the DeepSpeed ZeRO engine (separate from the
    ALST `sp_backend="deepspeed"` test). Distinct docker image, hence its own file/flag."""

    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    @parameterized.expand([2, 3])
    def test_ulysses_sp_deepspeed(self, zero_stage):  # 2 GPUs: native sp=2 under ZeRO-{2,3}
        script = self.test_scripts_folder / "test_accelerate_ulysses_sp.py"
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=2",
            "--num_machines=1",
            "--machine_rank=0",
            "--use_deepspeed",
            f"--zero_stage={zero_stage}",
            str(script),
            "--engine=deepspeed",
        ]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)
