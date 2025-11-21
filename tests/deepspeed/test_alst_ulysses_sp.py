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

from parameterized import parameterized

from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    path_in_accelerate_package,
    require_deepspeed,
    require_multi_device,
)
from accelerate.utils import patch_environment


@require_deepspeed
@require_multi_device
class DeepSpeedALSTUlyssesSPTest(TempDirTestCase):
    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    @parameterized.expand([2, 3])
    def test_deepspeed_alst_ulysses_sp(self, stage):
        self.test_file_path = self.test_scripts_folder / "test_ds_alst_ulysses_sp.py"
        world_size = 2
        cmd = [
            "accelerate",
            "launch",
            f"--num_processes={world_size}",
            "--num_machines=1",
            "--machine_rank=0",
            "--mixed_precision=bf16",
            "--use_deepspeed",
            f"--zero_stage={stage}",
            self.test_file_path,
            f"--output_dir={self.tmpdir}",
        ]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)
