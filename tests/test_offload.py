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
from accelerate.utils.versions import is_torch_version


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

        # NumPy has no FP8 dtypes either, so these go through the same int8-view path as
        # bfloat16 goes through int16. torch.randn(..., dtype=float8_*) isn't implemented, so
        # the tensors are produced the way they actually show up in practice: cast down from a
        # higher-precision tensor.
        if is_torch_version(">=", "2.1.0"):
            for name in ("float8_e4m3fn", "float8_e5m2"):
                if hasattr(torch, name):
                    dtypes.append(getattr(torch, name))

        for dtype in dtypes:
            is_fp8 = str(dtype).startswith("torch.float8_")
            weight = torch.randn(2, 3, dtype=torch.float32).to(dtype) if is_fp8 else torch.randn(2, 3, dtype=dtype)
            with TemporaryDirectory() as tmp_dir:
                index = offload_weight(weight, "weight", tmp_dir, {})
                weight_file = os.path.join(tmp_dir, "weight.dat")
                assert os.path.isfile(weight_file)
                assert index == {"weight": {"shape": [2, 3], "dtype": str(dtype).split(".")[1]}}

                new_weight = load_offloaded_weight(weight_file, index["weight"])
                assert new_weight.dtype == weight.dtype
                # Compare on raw bits: FP8/bfloat16 round-trip through an int view, and float
                # equality can't be trusted to catch a byte-level corruption anyway.
                int_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32}[weight.element_size()]
                assert torch.equal(weight.view(int_dtype), new_weight.view(int_dtype))

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
