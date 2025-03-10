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
import unittest

import torch

from accelerate import Accelerator
from accelerate.big_modeling import dispatch_model
from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    assert_exception,
    device_count,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
    require_huggingface_suite,
    require_multi_device,
    require_non_hpu,
    require_non_torch_xla,
    require_non_xpu,
    require_pippy,
    require_torchvision,
    run_first,
    torch_device,
)
from accelerate.utils import is_hpu_available, patch_environment


class MultiDeviceTester(unittest.TestCase):
    test_file_path = path_in_accelerate_package("test_utils", "scripts", "test_script.py")
    data_loop_file_path = path_in_accelerate_package("test_utils", "scripts", "test_distributed_data_loop.py")
    operation_file_path = path_in_accelerate_package("test_utils", "scripts", "test_ops.py")
    pippy_file_path = path_in_accelerate_package("test_utils", "scripts", "external_deps", "test_pippy.py")
    merge_weights_file_path = path_in_accelerate_package("test_utils", "scripts", "test_merge_weights.py")

    @run_first
    @require_multi_device
    def test_multi_device(self):
        print(f"Found {device_count} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [self.test_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)

    @run_first
    @require_multi_device
    def test_multi_device_ops(self):
        print(f"Found {device_count} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [self.operation_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)

    @run_first
    @require_multi_device
    def test_pad_across_processes(self):
        print(f"Found {device_count} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [inspect.getfile(self.__class__)]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)

    @run_first
    @require_non_hpu  # Synapse detected a device critical error that requires a restart
    @require_multi_device
    def test_multi_device_merge_fsdp_weights(self):
        print(f"Found {device_count} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [self.merge_weights_file_path]

        env_kwargs = dict(omp_num_threads=1)
        with patch_environment(**env_kwargs):
            execute_subprocess_async(cmd)

    @run_first
    @require_non_torch_xla
    @require_multi_device
    def test_distributed_data_loop(self):
        """
        This TestCase checks the behaviour that occurs during distributed training or evaluation,
        when the batch size does not evenly divide the dataset size.
        """
        print(f"Found {device_count} devices, using 2 devices only")
        cmd = get_launch_command(num_processes=2) + [self.data_loop_file_path]

        env_kwargs = dict(omp_num_threads=1)
        if torch_device == "xpu":
            env_kwargs.update(ze_affinity_mask="0,1")
        elif torch_device == "npu":
            env_kwargs.update(ascend_rt_visible_devices="0,1")
        elif torch_device == "mlu":
            env_kwargs.update(mlu_visible_devices="0,1")
        elif torch_device == "sdaa":
            env_kwargs.update(sdaa_visible_devices="0,1")
        else:
            env_kwargs.update(cuda_visible_devices="0,1")

        with patch_environment(**env_kwargs):
            execute_subprocess_async(cmd)

    @run_first
    @require_pippy
    @require_non_xpu
    @require_torchvision
    @require_multi_device
    @require_huggingface_suite
    def test_pippy(self):
        """
        Checks the integration with the pippy framework
        """
        print(f"Found {device_count} devices")
        cmd = get_launch_command(multi_gpu=True, num_processes=device_count) + [self.pippy_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)


if __name__ == "__main__":
    accelerator = Accelerator()
    shape = (accelerator.state.process_index + 2, 10)
    tensor = torch.randint(0, 10, shape).to(accelerator.device)

    error_msg = ""

    tensor1 = accelerator.pad_across_processes(tensor)
    if tensor1.shape[0] != accelerator.state.num_processes + 1:
        error_msg += f"Found shape {tensor1.shape} but should have {accelerator.state.num_processes + 1} at dim 0."
    index = accelerator.state.process_index + 2
    if not torch.equal(tensor1[:index], tensor):
        error_msg += "Tensors have different values."
    if not torch.all(tensor1[index:] == 0):
        error_msg += "Padding was not done with the right value (0)."

    tensor2 = accelerator.pad_across_processes(tensor.clone(), pad_first=True)
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

    if is_hpu_available():
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": 0}
    else:
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": 1}

    model = ModelForTest()
    dispatch_model(model, device_map=device_map)
    with assert_exception(ValueError, "You can't train a model that has been loaded with"):
        model = accelerator.prepare_model(model)
