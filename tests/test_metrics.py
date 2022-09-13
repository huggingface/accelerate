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

import inspect
import os
import unittest

import torch

import accelerate
from accelerate import debug_launcher
from accelerate.test_utils import (
    execute_subprocess_async,
    require_cpu,
    require_huggingface_suite,
    require_multi_gpu,
    require_single_gpu,
    require_torch_min_version,
)
from accelerate.utils import get_launch_prefix, patch_environment


@require_huggingface_suite
@require_torch_min_version(version="1.8.0")
class MetricTester(unittest.TestCase):
    def setUp(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_file_path = os.path.sep.join(
            mod_file.split(os.path.sep)[:-1] + ["scripts", "external_deps", "test_metrics.py"]
        )

        from accelerate.test_utils.scripts.external_deps import test_metrics  # noqa: F401

        self.test_metrics = test_metrics

    @require_cpu
    def test_metric_cpu_noop(self):
        debug_launcher(self.test_metrics.main, num_processes=1)

    @require_cpu
    def test_metric_cpu_multi(self):
        debug_launcher(self.test_metrics.main)

    @require_single_gpu
    def test_metric_gpu(self):
        self.test_metrics.main()

    @require_multi_gpu
    def test_metric_gpu_multi(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        cmd = get_launch_prefix() + [f"--nproc_per_node={torch.cuda.device_count()}", self.test_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())
