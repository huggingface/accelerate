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

from accelerate.utils import (
    OffloadedWeightsLoader,
    extract_submodules_state_dict,
    load_offloaded_weight,
    offload_state_dict,
    offload_weight,
)


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
        model = ModelForTest()
        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, model.state_dict())
            index_file = os.path.join(tmp_dir, "index.json")
            assert os.path.isfile(index_file)
            # TODO: add tests on what is inside the index

            for key in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
                weight_file = os.path.join(tmp_dir, f"{key}.dat")
                assert os.path.isfile(weight_file)
                # TODO: add tests on the fact weights are properly loaded

    def test_offload_weight(self):
        dtypes = [torch.float16, torch.float32, torch.bfloat16]

        for dtype in dtypes:
            weight = torch.randn(2, 3, dtype=dtype)
            with TemporaryDirectory() as tmp_dir:
                index = offload_weight(weight, "weight", tmp_dir, {})
                weight_file = os.path.join(tmp_dir, "weight.dat")
                assert os.path.isfile(weight_file)
                assert index == {"weight": {"shape": [2, 3], "dtype": str(dtype).split(".")[1]}}

                new_weight = load_offloaded_weight(weight_file, index["weight"])
                assert torch.equal(weight, new_weight)

    def test_offload_weights_loader(self):
        model = ModelForTest()
        state_dict = model.state_dict()
        cpu_part = {k: v for k, v in state_dict.items() if "linear2" not in k}
        disk_part = {k: v for k, v in state_dict.items() if "linear2" in k}

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, disk_part)
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            assert sorted(weight_map) == sorted(state_dict.keys())
            for key, param in state_dict.items():
                assert torch.allclose(param, weight_map[key])

        cpu_part = {k: v for k, v in state_dict.items() if "weight" in k}
        disk_part = {k: v for k, v in state_dict.items() if "weight" not in k}

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, disk_part)
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            assert sorted(weight_map) == sorted(state_dict.keys())
            for key, param in state_dict.items():
                assert torch.allclose(param, weight_map[key])

        with TemporaryDirectory() as tmp_dir:
            offload_state_dict(tmp_dir, state_dict)
            # Duplicates are removed
            weight_map = OffloadedWeightsLoader(state_dict=cpu_part, save_folder=tmp_dir)

            # Every key is there with the right value
            assert sorted(weight_map) == sorted(state_dict.keys())
            for key, param in state_dict.items():
                assert torch.allclose(param, weight_map[key])

    def test_extract_submodules_state_dict(self):
        state_dict = {"a.1": 0, "a.10": 1, "a.2": 2}
        extracted = extract_submodules_state_dict(state_dict, ["a.1", "a.2"])
        assert extracted == {"a.1": 0, "a.2": 2}

        state_dict = {"a.1.a": 0, "a.10.a": 1, "a.2.a": 2}
        extracted = extract_submodules_state_dict(state_dict, ["a.1", "a.2"])
        assert extracted == {"a.1.a": 0, "a.2.a": 2}
