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

import inspect
import os
import sys
import unittest

import torch

import accelerate
from accelerate import Accelerator
from accelerate.utils import get_launch_prefix, patch_environment
from accelerate.test_utils import execute_subprocess_async, require_multi_gpu


class MultiGPUTester(unittest.TestCase):
    def setUp(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["test_script.py"])

    @require_multi_gpu
    def test_multi_gpu(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        distributed_args = f"""
        --nproc_per_node={torch.cuda.device_count()}
        {self.test_file_path}
        """.split()
        cmd = get_launch_prefix() + distributed_args
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())

    @require_multi_gpu
    def test_pad_across_processes(self):
        distributed_args = f"""
            --nproc_per_node={torch.cuda.device_count()}
            {inspect.getfile(self.__class__)}
        """.split()
        cmd = get_launch_prefix() + distributed_args
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())


if __name__ == "__main__":
    accelerator = Accelerator()
    shape = (accelerator.state.process_index + 2, 10)
    tensor = torch.randint(0, 10, shape).to(accelerator.device)

    error_msg = ""

    tensor1 = accelerator.pad_across_processes(tensor)
    if tensor1.shape[0] != accelerator.state.num_processes + 1:
        error_msg += f"Found shape {tensor1.shape} but should have {accelerator.state.num_processes + 1} at dim 0."
    if not torch.equal(tensor1[: accelerator.state.process_index + 2], tensor):
        error_msg += "Tensors have different values."
    if not torch.all(tensor1[accelerator.state.process_index + 2 :] == 0):
        error_msg += "Padding was not done with the right value (0)."

    tensor2 = accelerator.pad_across_processes(tensor, pad_first=True)
    if tensor2.shape[0] != accelerator.state.num_processes + 1:
        error_msg += f"Found shape {tensor2.shape} but should have {accelerator.state.num_processes + 1} at dim 0."
    index = accelerator.state.num_processes - accelerator.state.process_index - 1
    if not torch.equal(tensor2[index:], tensor):
        error_msg += "Tensors have different values."
    if not torch.all(tensor2[:index] == 0):
        error_msg += "Padding was not done with the right value (0)."

    # Raise error at the end to make sure we don't stop at the first failure.
    if len(error_msg) > 0:
        raise ValueError(error_msg)
