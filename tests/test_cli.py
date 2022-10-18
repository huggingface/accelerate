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

import torch
import os
import unittest
from tempfile import TemporaryDirectory
import inspect
from accelerate.test_utils import execute_subprocess_async
import accelerate


class AccelerateLauncherTester(unittest.TestCase):
    mod_file = inspect.getfile(accelerate.test_utils)
    test_file_path = os.path.sep.join(
        mod_file.split(os.path.sep)[:-1] + ["scripts", "test_cli.py"]
    )
    base_cmd = ["accelerate", "launch"]

    def test_no_config(self):
        execute_subprocess_async(self.base_cmd + [self.test_file_path], env=os.environ.copy())