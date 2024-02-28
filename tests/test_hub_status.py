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

from accelerate.test_utils import is_hub_online


class HooksModelTester(unittest.TestCase):
    "Simple tester that checks if the Hub is online or not"

    def test_hub_online(self):
        self.assertTrue(
            is_hub_online(),
            "Hub is offline! This test will fail until the hub is back online. Relevent tests will be skipped.",
        )
