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
from torch import nn

from accelerate.test_utils import require_cuda
from accelerate.utils.memory import find_executable_batch_size, release_memory


def raise_fake_out_of_memory():
    raise RuntimeError("CUDA out of memory.")


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class MemoryTest(unittest.TestCase):
    def test_memory_implicit(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size != 8:
                raise_fake_out_of_memory()

        mock_training_loop_function()
        self.assertListEqual(batch_sizes, [128, 64, 32, 16, 8])

    def test_memory_explicit(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size, arg1):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size != 8:
                raise_fake_out_of_memory()
            return batch_size, arg1

        bs, arg1 = mock_training_loop_function("hello")
        self.assertListEqual(batch_sizes, [128, 64, 32, 16, 8])
        self.assertListEqual([bs, arg1], [8, "hello"])

    def test_start_zero(self):
        @find_executable_batch_size(starting_batch_size=0)
        def mock_training_loop_function(batch_size):
            pass

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            self.assertIn("No executable batch size found, reached zero.", cm.exception.args[0])

    def test_approach_zero(self):
        @find_executable_batch_size(starting_batch_size=16)
        def mock_training_loop_function(batch_size):
            if batch_size > 0:
                raise_fake_out_of_memory()
            pass

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            self.assertIn("No executable batch size found, reached zero.", cm.exception.args[0])

    def test_verbose_guard(self):
        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size, arg1, arg2):
            if batch_size != 8:
                raise raise_fake_out_of_memory()

        with self.assertRaises(TypeError) as cm:
            mock_training_loop_function(128, "hello", "world")
            self.assertIn("Batch size was passed into `f`", cm.exception.args[0])
            self.assertIn("`f(arg1='hello', arg2='world')", cm.exception.args[0])

    def test_any_other_error(self):
        @find_executable_batch_size(starting_batch_size=16)
        def mock_training_loop_function(batch_size):
            raise ValueError("Oops, we had an error!")

        with self.assertRaises(ValueError) as cm:
            mock_training_loop_function()
            self.assertIn("Oops, we had an error!", cm.exception.args[0])

    @require_cuda
    def test_release_memory(self):
        self.assertEqual(torch.cuda.memory_allocated(), 0)
        model = ModelForTest()
        model.cuda()
        self.assertGreater(torch.cuda.memory_allocated(), 0)
        model = release_memory(model)
        self.assertEqual(torch.cuda.memory_allocated(), 0)
