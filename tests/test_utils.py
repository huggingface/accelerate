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
from collections import namedtuple

import torch

from accelerate.utils import send_to_device


TestNamedTuple = namedtuple("TestNamedTuple", "a b c")


class UtilsTester(unittest.TestCase):
    def test_send_to_device(self):
        tensor = torch.randn(5, 2)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        result1 = send_to_device(tensor, device)
        self.assertTrue(torch.equal(result1.cpu(), tensor))

        result2 = send_to_device((tensor, [tensor, tensor], 1), device)
        self.assertIsInstance(result2, tuple)
        self.assertTrue(torch.equal(result2[0].cpu(), tensor))
        self.assertIsInstance(result2[1], list)
        self.assertTrue(torch.equal(result2[1][0].cpu(), tensor))
        self.assertTrue(torch.equal(result2[1][1].cpu(), tensor))
        self.assertEqual(result2[2], 1)

        result2 = send_to_device({"a": tensor, "b": [tensor, tensor], "c": 1}, device)
        self.assertIsInstance(result2, dict)
        self.assertTrue(torch.equal(result2["a"].cpu(), tensor))
        self.assertIsInstance(result2["b"], list)
        self.assertTrue(torch.equal(result2["b"][0].cpu(), tensor))
        self.assertTrue(torch.equal(result2["b"][1].cpu(), tensor))
        self.assertEqual(result2["c"], 1)

        result3 = send_to_device(TestNamedTuple(a=tensor, b=[tensor, tensor], c=1), device)
        self.assertIsInstance(result3, TestNamedTuple)
        self.assertTrue(torch.equal(result3.a.cpu(), tensor))
        self.assertIsInstance(result3.b, list)
        self.assertTrue(torch.equal(result3.b[0].cpu(), tensor))
        self.assertTrue(torch.equal(result3.b[1].cpu(), tensor))
        self.assertEqual(result3.c, 1)
