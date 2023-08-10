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
import unittest

import torch

import accelerate
from accelerate import Accelerator
from accelerate.big_modeling import dispatch_model
from accelerate.test_utils import assert_exception, execute_subprocess_async, require_multi_gpu
from accelerate.utils import patch_environment


class MultiGPUTester(unittest.TestCase):
    def setUp(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "test_script.py"])
        self.data_loop_file_path = os.path.sep.join(
            mod_file.split(os.path.sep)[:-1] + ["scripts", "test_distributed_data_loop.py"]
        )
        self.operation_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "test_ops.py"])

    @require_multi_gpu
    def test_multi_gpu(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        cmd = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}", self.test_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())

    @require_multi_gpu
    def test_multi_gpu_ops(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        cmd = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}", self.operation_file_path]
        print(f"Command: {cmd}")
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())

    @require_multi_gpu
    def test_pad_across_processes(self):
        cmd = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd, env=os.environ.copy())

    @require_multi_gpu
    def test_distributed_data_loop(self):
        """
        This TestCase checks the behaviour that occurs during distributed training or evaluation,
        when the batch size does not evenly divide the dataset size.
        """
        print(f"Found {torch.cuda.device_count()} devices, using 2 devices only")
        cmd = ["torchrun", f"--nproc_per_node={torch.cuda.device_count()}", self.data_loop_file_path]
        with patch_environment(omp_num_threads=1, cuda_visible_devices="0,1"):
            execute_subprocess_async(cmd, env=os.environ.copy())

    @require_multi_gpu
    def test_notebook_launcher(self):
        """
        This test checks that the `notebook_launcher` will be able to intialize
        a `PartialState` without issue
        """
        cmd = [
            "python",
            "-m",
            "accelerate.test_utils.scripts.test_notebook",
            "--num_processes",
            str(torch.cuda.device_count()),
        ]
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

    # Check device_map
    accelerator.print("Test `device_map` cannot be prepared.")

    class ModelForTest(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(3, 4)
            self.batchnorm = torch.nn.BatchNorm1d(4)
            self.linear2 = torch.nn.Linear(4, 5)

        def forward(self, x):
            return self.linear2(self.batchnorm(self.linear1(x)))

    device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": 1}
    model = ModelForTest()
    dispatch_model(model, device_map=device_map)
    with assert_exception(ValueError, "You can't train a model that has been loaded with"):
        model = accelerator.prepare_model(model)
