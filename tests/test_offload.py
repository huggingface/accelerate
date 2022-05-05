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

import os
import unittest
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from accelerate.utils import OffloadedWeightsLoader, offload_state_dict


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class OffloadTester(unittest.TestCase):
    def test_offload_state_dict(self):
        from tempfile import TemporaryDirectory

        model = ModelForTest()
        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, model.state_dict())
            index_file = os.path.join(tmp_dir, "index.json")
            self.assertTrue(os.path.isfile(index_file))
            # TODO: add tests on what is inside the index

            for key in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
                weight_file = os.path.join(tmp_dir, f"{key}.dat")
                self.assertTrue(os.path.isfile(weight_file))
                # TODO: add tests on the fact weights are properly loaded

    def test_offload_weights_loader(self):
        model = ModelForTest()
        state_dict = model.state_dict()
        cpu_part = {k: v for k, v in state_dict.items() if "linear2" not in k}
        disk_part = {k: v for k, v in state_dict.items() if "linear2" in k}

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, disk_part)
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            self.assertEqual(sorted(weight_map), sorted(state_dict.keys()))
            for key, param in state_dict.items():
                self.assertTrue(torch.allclose(param, weight_map[key]))

        cpu_part = {k: v for k, v in state_dict.items() if "weight" in k}
        disk_part = {k: v for k, v in state_dict.items() if "weight" not in k}

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, disk_part)
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            self.assertEqual(sorted(weight_map), sorted(state_dict.keys()))
            for key, param in state_dict.items():
                self.assertTrue(torch.allclose(param, weight_map[key]))

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, state_dict)
            # Duplicates are removed
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            self.assertEqual(sorted(weight_map), sorted(state_dict.keys()))
            for key, param in state_dict.items():
                self.assertTrue(torch.allclose(param, weight_map[key]))
