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

import unittest

from parameterized import parameterized

from accelerate import debug_launcher
from accelerate.test_utils import require_cpu, test_ops, test_script
from accelerate.test_utils.scripts.test_distributed_data_loop import (
    test_iterable_shard_metric_samples as iterable_shard_metric_samples_test,
)


@require_cpu
class MultiCPUTester(unittest.TestCase):
    def test_cpu(self):
        debug_launcher(test_script.main)

    def test_ops(self):
        debug_launcher(test_ops.main)

    @parameterized.expand([(2,), (4,)])
    def test_iterable_shard_metric_samples(self, num_processes):
        debug_launcher(iterable_shard_metric_samples_test, num_processes=num_processes)
