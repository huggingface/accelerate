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

import torch
import torch.nn as nn

from accelerate.test_utils import require_cuda, require_multi_gpu
from accelerate.utils.modeling import (
    compute_module_sizes,
    find_tied_parameters,
    named_module_tensors,
    set_module_tensor_to_device,
)


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class ModelingUtilsTester(unittest.TestCase):
    def check_set_module_tensor_for_device(self, model, device1, device2):
        self.assertEqual(model.linear1.weight.device, torch.device(device1))

        with self.subTest("Access by submodule and direct name for a parameter"):
            set_module_tensor_to_device(model.linear1, "weight", device2)
            self.assertEqual(model.linear1.weight.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model.linear1, "weight", device1)

                set_module_tensor_to_device(model.linear1, "weight", device1, value=torch.randn(4, 3))
            else:
                set_module_tensor_to_device(model.linear1, "weight", device1)
            self.assertEqual(model.linear1.weight.device, torch.device(device1))

        with self.subTest("Access by module and full name for a parameter"):
            set_module_tensor_to_device(model, "linear1.weight", device2)
            self.assertEqual(model.linear1.weight.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model, "linear1.weight", device1)
                set_module_tensor_to_device(model, "linear1.weight", device1, value=torch.randn(4, 3))
            else:
                set_module_tensor_to_device(model, "linear1.weight", device1)
            self.assertEqual(model.linear1.weight.device, torch.device(device1))

        self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

        with self.subTest("Access by submodule and direct name for a buffer"):
            set_module_tensor_to_device(model.batchnorm, "running_mean", device2)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model.batchnorm, "running_mean", device1)
                set_module_tensor_to_device(model.batchnorm, "running_mean", device1, value=torch.randn(4))
            else:
                set_module_tensor_to_device(model.batchnorm, "running_mean", device1)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

        with self.subTest("Access by module and full name for a parameter"):
            set_module_tensor_to_device(model, "batchnorm.running_mean", device2)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on CPU
                    set_module_tensor_to_device(model, "batchnorm.running_mean", device1)

                set_module_tensor_to_device(model, "batchnorm.running_mean", device1, value=torch.randn(4))
            else:
                set_module_tensor_to_device(model, "batchnorm.running_mean", device1)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

    def test_set_module_tensor_to_meta_and_cpu(self):
        model = ModelForTest()
        self.check_set_module_tensor_for_device(model, "cpu", "meta")

    @require_cuda
    def test_set_module_tensor_to_cpu_and_gpu(self):
        model = ModelForTest()
        self.check_set_module_tensor_for_device(model, "cpu", 0)

    @require_cuda
    def test_set_module_tensor_to_meta_and_gpu(self):
        model = ModelForTest().to(0)
        self.check_set_module_tensor_for_device(model, 0, "meta")

    @require_multi_gpu
    def test_set_module_tensor_between_gpus(self):
        model = ModelForTest().to(0)
        self.check_set_module_tensor_for_device(model, 0, 1)

    def test_named_tensors(self):
        model = nn.BatchNorm1d(4)
        named_tensors = named_module_tensors(model)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"],
        )

        named_tensors = named_module_tensors(model, include_buffers=False)
        self.assertListEqual([name for name, _ in named_tensors], ["weight", "bias"])

        model = ModelForTest()
        named_tensors = named_module_tensors(model)
        self.assertListEqual([name for name, _ in named_tensors], [])

        named_tensors = named_module_tensors(model, recurse=True)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            [
                "linear1.weight",
                "linear1.bias",
                "batchnorm.weight",
                "batchnorm.bias",
                "linear2.weight",
                "linear2.bias",
                "batchnorm.running_mean",
                "batchnorm.running_var",
                "batchnorm.num_batches_tracked",
            ],
        )

        named_tensors = named_module_tensors(model, include_buffers=False, recurse=True)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            ["linear1.weight", "linear1.bias", "batchnorm.weight", "batchnorm.bias", "linear2.weight", "linear2.bias"],
        )

    def test_find_tied_parameters(self):
        model = ModelForTest()
        self.assertDictEqual(find_tied_parameters(model), {})
        model.linear2.weight = model.linear1.weight
        self.assertDictEqual(find_tied_parameters(model), {"linear1.weight": "linear2.weight"})

    def test_compute_module_sizes(self):
        model = ModelForTest()
        expected_sizes = {"": 236, "linear1": 64, "linear1.weight": 48, "linear1.bias": 16}
        expected_sizes.update({"linear2": 100, "linear2.weight": 80, "linear2.bias": 20})
        expected_sizes.update({"batchnorm": 72, "batchnorm.weight": 16, "batchnorm.bias": 16})
        expected_sizes.update(
            {"batchnorm.running_mean": 16, "batchnorm.running_var": 16, "batchnorm.num_batches_tracked": 8}
        )

        module_sizes = compute_module_sizes(model)
        self.assertDictEqual(module_sizes, expected_sizes)

        model.half()
        expected_sizes = {k: s // 2 for k, s in expected_sizes.items()}
        # This one is not converted to half.
        expected_sizes["batchnorm.num_batches_tracked"] = 8
        # This impacts batchnorm and total
        expected_sizes["batchnorm"] += 4
        expected_sizes[""] += 4

        module_sizes = compute_module_sizes(model)
        self.assertDictEqual(module_sizes, expected_sizes)
