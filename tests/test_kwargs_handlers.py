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
from dataclasses import dataclass

import torch

from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from accelerate.state import AcceleratorState
from accelerate.test_utils import execute_subprocess_async, require_cuda, require_multi_gpu
from accelerate.utils import KwargsHandler, get_launch_prefix


@dataclass
class MockClass(KwargsHandler):
    a: int = 0
    b: bool = False
    c: float = 3.0


class DataLoaderTester(unittest.TestCase):
    def test_kwargs_handler(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        self.assertDictEqual(MockClass().to_kwargs(), {})
        self.assertDictEqual(MockClass(a=2).to_kwargs(), {"a": 2})
        self.assertDictEqual(MockClass(a=2, b=True).to_kwargs(), {"a": 2, "b": True})
        self.assertDictEqual(MockClass(a=2, c=2.25).to_kwargs(), {"a": 2, "c": 2.25})

    @require_cuda
    def test_grad_scaler_kwargs(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        scaler_handler = GradScalerKwargs(init_scale=1024, growth_factor=2)
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[scaler_handler])
        print(accelerator.use_fp16)
        scaler = accelerator.scaler

        # Check the kwargs have been applied
        self.assertEqual(scaler._init_scale, 1024.0)
        self.assertEqual(scaler._growth_factor, 2.0)

        # Check the other values are at the default
        self.assertEqual(scaler._backoff_factor, 0.5)
        self.assertEqual(scaler._growth_interval, 2000)
        self.assertEqual(scaler._enabled, True)

    @require_multi_gpu
    def test_ddp_kwargs(self):
        cmd = get_launch_prefix()
        cmd += [f"--nproc_per_node={torch.cuda.device_count()}", inspect.getfile(self.__class__)]
        execute_subprocess_async(cmd, env=os.environ.copy())


if __name__ == "__main__":
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    model = torch.nn.Linear(100, 200)
    model = accelerator.prepare(model)

    # Check the values changed in kwargs
    error_msg = ""
    observed_bucket_cap_map = model.bucket_bytes_cap // (1024 * 1024)
    if observed_bucket_cap_map != 15:
        error_msg += f"Kwargs badly passed, should have `15` but found {observed_bucket_cap_map}.\n"
    if model.find_unused_parameters is not True:
        error_msg += f"Kwargs badly passed, should have `True` but found {model.find_unused_parameters}.\n"

    # Check the values of the defaults
    if model.dim != 0:
        error_msg += f"Default value not respected, should have `0` but found {model.dim}.\n"
    if model.broadcast_buffers is not True:
        error_msg += f"Default value not respected, should have `True` but found {model.broadcast_buffers}.\n"
    if model.gradient_as_bucket_view is not False:
        error_msg += f"Default value not respected, should have `False` but found {model.gradient_as_bucket_view}.\n"

    # Raise error at the end to make sure we don't stop at the first failure.
    if len(error_msg) > 0:
        raise ValueError(error_msg)
