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
import subprocess
import sys

from accelerate.test_utils import require_transformer_engine
from accelerate.test_utils.testing import TempDirTestCase, require_import_timer
from accelerate.utils import is_import_timer_available


if is_import_timer_available():
    from import_timer import calculate_total_time, read_import_profile
    from import_timer.core import get_paths_above_threshold, sort_nodes_by_total_time


def convert_list_to_string(data):
    end_result = ""
    arrow_right = "->"
    for path in data:
        end_result += f"{arrow_right.join(path[0])} {path[1]:.3f}s\n"
    return end_result


def run_import_time(command: str):
    output = subprocess.run([sys.executable, "-X", "importtime", "-c", command], capture_output=True, text=True)
    return output.stderr


@require_import_timer
class ImportSpeedTester(TempDirTestCase):
    """
    Test suite which checks if imports have seen slowdowns
    based on a particular baseline.

    If the error messages are not clear enough to get a
    full view of what is slowing things down (or to
    figure out how deep the initial depth should be),
    please view the profile with the `tuna` framework:
    `tuna import.log`.
    """

    clear_on_setup = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        output = run_import_time("import torch")
        data = read_import_profile(output)
        total_time = calculate_total_time(data)
        cls.pytorch_time = total_time

    def test_base_import(self):
        output = run_import_time("import accelerate")
        data = read_import_profile(output)
        total_time = calculate_total_time(data)
        pct_more = (total_time - self.pytorch_time) / self.pytorch_time * 100
        # Base import should never be more than 20% slower than raw torch import
        err_msg = f"Base import is more than 20% slower than raw torch import ({pct_more:.2f}%), please check the attached `tuna` profile:\n"
        sorted_data = sort_nodes_by_total_time(data)
        paths_above_threshold = get_paths_above_threshold(sorted_data, 0.05, max_depth=7)
        err_msg += f"\n{convert_list_to_string(paths_above_threshold)}"
        self.assertLess(pct_more, 20, err_msg)

    def test_cli_import(self):
        output = run_import_time("from accelerate.commands.launch import launch_command_parser")
        data = read_import_profile(output)
        total_time = calculate_total_time(data)
        pct_more = (total_time - self.pytorch_time) / self.pytorch_time * 100
        # Base import should never be more than 20% slower than raw torch import
        err_msg = f"Base import is more than 20% slower than raw torch import ({pct_more:.2f}%), please check the attached `tuna` profile:\n"
        sorted_data = sort_nodes_by_total_time(data)
        paths_above_threshold = get_paths_above_threshold(sorted_data, 0.05, max_depth=7)
        err_msg += f"\n{convert_list_to_string(paths_above_threshold)}"
        self.assertLess(pct_more, 20, err_msg)


@require_transformer_engine
class LazyImportTester(TempDirTestCase):
    """
    Test suite which checks if specific packages are lazy-loaded.

    Eager-import will trigger circular import in some case,
    e.g. in huggingface/accelerate#3056.
    """

    def test_te_import(self):
        output = run_import_time("import accelerate, accelerate.utils.transformer_engine")

        self.assertFalse(" transformer_engine" in output, "`transformer_engine` should not be imported on import")
