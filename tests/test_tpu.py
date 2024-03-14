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

import os
import sys
import unittest

from accelerate.test_utils import execute_subprocess_async, path_in_accelerate_package, require_tpu


class MultiTPUTester(unittest.TestCase):
    test_file_path = path_in_accelerate_package("test_utils", "scripts", "test_script.py")
    test_dir = os.path.dirname(__file__)

    @require_tpu
    def test_tpu(self):
        distributed_args = f"""
            {self.test_dir}/xla_spawn.py
            --num_cores 8
            {self.test_file_path}
        """.split()
        cmd = [sys.executable] + distributed_args
        execute_subprocess_async(cmd)
