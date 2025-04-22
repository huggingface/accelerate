# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import warnings
import os
import torch
from unittest.mock import patch, MagicMock

from accelerate.state import PartialState
from accelerate.utils import gather_object, patch_environment
from accelerate.test_utils import (
    require_cuda,
    require_multi_device,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
)


class TestPartialStateDevice(unittest.TestCase):
    """
    Test cases for proper device management in PartialState to prevent duplicate GPU usage
    in distributed processing.
    """
    
    test_script_path = path_in_accelerate_package("test_utils", "scripts", "test_partial_state_device.py")
    
    def setUp(self):
        self.has_multiple_gpus = torch.cuda.is_available() and torch.cuda.device_count() > 1
    
    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.device_count() <= 1,
                    "Test requires multiple GPUs")
    def test_prevent_duplicate_gpu_usage(self):
        """
        Tests that when there are more processes than available GPUs, the code properly
        handles device assignment to prevent duplicate GPU usage.
        """
        cuda_device_count = torch.cuda.device_count()
        # Use more processes than available GPUs
        num_processes = cuda_device_count + 1
        
        cmd = get_launch_command(num_processes=num_processes) + [self.test_script_path]
        
        # Make only cuda_device_count GPUs visible
        visible_devices = ",".join(str(i) for i in range(cuda_device_count))
        with patch_environment(cuda_visible_devices=visible_devices):
            # This should execute without raising NCCL errors due to duplicate GPU usage
            result = execute_subprocess_async(cmd, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, f"Process failed with error: {result.stderr.decode('utf-8')}")
    
    @require_cuda
    def test_device_assignment_warning(self):
        """
        Tests that a warning is raised when PartialState is initialized with more processes 
        than available GPUs.
        """
        # Mock CUDA device count to return 1
        with patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.is_available", return_value=True), \
             patch.dict(os.environ, {
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "2",
                "RANK": "0",
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29500",
                "ACCELERATE_DISTRIBUTED_TYPE": "MULTI_GPU"
            }), patch("torch.distributed.init_process_group") as mock_init, \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.distributed.get_backend", return_value="nccl"):
            
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Initialize PartialState
                state = PartialState()
                
                # Check if warning was raised
                warning_messages = [str(warning.message) for warning in w]
                self.assertTrue(
                    any("fewer GPUs (1) than processes (2)" in msg for msg in warning_messages),
                    f"Expected warning about fewer GPUs than processes was not raised. Got warnings: {warning_messages}"
                )
    
    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.device_count() <= 1,
                    "Test requires multiple GPUs")
    def test_padding_with_extra_samples(self):
        """
        Tests the padding behavior when there are extra samples to distribute.
        This test verifies the fix from PR #3518 where padding calculation was modified
        to properly handle extra samples.
        """
        # Use 3 processes to test padding with extra samples
        num_processes = 3
        cmd = get_launch_command(num_processes=num_processes) + [self.test_script_path, "--test-padding"]
        
        result = execute_subprocess_async(cmd, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, f"Process failed with error: {result.stderr.decode('utf-8')}")


if __name__ == "__main__":
    unittest.main()
