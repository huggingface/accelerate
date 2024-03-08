# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from accelerate import debug_launcher
from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    device_count,
    execute_subprocess_async,
    path_in_accelerate_package,
    require_cpu,
    require_huggingface_suite,
    require_multi_device,
    require_single_device,
)
from accelerate.utils import patch_environment


@require_huggingface_suite
class MetricTester(unittest.TestCase):
    def setUp(self):
        self.test_file_path = path_in_accelerate_package("test_utils", "scripts", "external_deps", "test_metrics.py")

        from accelerate.test_utils.scripts.external_deps import test_metrics  # noqa: F401

        self.test_metrics = test_metrics

    @require_cpu
    def test_metric_cpu_noop(self):
        debug_launcher(self.test_metrics.main, num_processes=1)

    @require_cpu
    def test_metric_cpu_multi(self):
        debug_launcher(self.test_metrics.main)

    @require_single_device
    def test_metric_accelerator(self):
        self.test_metrics.main()

    @require_multi_device
    def test_metric_accelerator_multi(self):
        print(f"Found {device_count} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [self.test_file_path]
        with patch_environment(omp_num_threads=1, ACCELERATE_LOG_LEVEL="INFO"):
            execute_subprocess_async(cmd)
