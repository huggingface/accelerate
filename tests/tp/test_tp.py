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


from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
    require_multi_device,
    require_non_torch_xla,
    require_tp,
    require_transformers,
    run_first,
    slow,
)
from accelerate.utils import patch_environment


@require_non_torch_xla
@require_tp
@require_multi_device
@require_transformers
@slow
class TPIntegrationTest(TempDirTestCase):
    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    def setUp(self):
        super().setUp()
        self.test_tp_size = 2
        self.model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.batch_size = 1
        from accelerate.utils import set_seed

        set_seed(42)

    @run_first
    def test_working_of_tp(self):
        self.test_file_path = self.test_scripts_folder / "test_performance.py"
        cmd = get_launch_command(
            num_processes=self.test_tp_size, num_machines=1, machine_rank=0, use_tp=True, tp_size=self.test_tp_size
        )
        cmd.extend(
            [
                self.test_file_path,
                f"--output_dir={self.tmpdir}",
                f"--model_name_or_path={self.model_name_or_path}",
                "--add_pad_token=true",
            ]
        )
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)
