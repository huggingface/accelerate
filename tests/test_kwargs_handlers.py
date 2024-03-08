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
from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    execute_subprocess_async,
    require_multi_device,
    require_non_cpu,
    require_non_xpu,
)
from accelerate.utils import AutocastKwargs, KwargsHandler, TorchDynamoPlugin, clear_environment
from accelerate.utils.dataclasses import DistributedType


@dataclass
class MockClass(KwargsHandler):
    a: int = 0
    b: bool = False
    c: float = 3.0


class KwargsHandlerTester(unittest.TestCase):
    def test_kwargs_handler(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        assert MockClass().to_kwargs() == {}
        assert MockClass(a=2).to_kwargs() == {"a": 2}
        assert MockClass(a=2, b=True).to_kwargs() == {"a": 2, "b": True}
        assert MockClass(a=2, c=2.25).to_kwargs() == {"a": 2, "c": 2.25}

    @require_non_cpu
    @require_non_xpu
    def test_grad_scaler_kwargs(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        scaler_handler = GradScalerKwargs(init_scale=1024, growth_factor=2)
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[scaler_handler])
        print(accelerator.use_fp16)
        scaler = accelerator.scaler

        # Check the kwargs have been applied
        assert scaler._init_scale == 1024.0
        assert scaler._growth_factor == 2.0

        # Check the other values are at the default
        assert scaler._backoff_factor == 0.5
        assert scaler._growth_interval == 2000
        assert scaler._enabled is True

    @require_multi_device
    def test_ddp_kwargs(self):
        cmd = DEFAULT_LAUNCH_COMMAND + [inspect.getfile(self.__class__)]
        execute_subprocess_async(cmd)

    @require_non_cpu
    def test_autocast_kwargs(self):
        kwargs = AutocastKwargs(enabled=False)
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16")

        a_float32 = torch.rand((8, 8), device=accelerator.device)
        b_float32 = torch.rand((8, 8), device=accelerator.device)
        c_float32 = torch.rand((8, 8), device=accelerator.device)
        d_float32 = torch.rand((8, 8), device=accelerator.device)

        with accelerator.autocast():
            e_float16 = torch.mm(a_float32, b_float32)
            assert e_float16.dtype == torch.float16

            with accelerator.autocast(autocast_handler=kwargs):
                # Convert e_float16 to float32
                f_float32 = torch.mm(c_float32, e_float16.float())
                assert f_float32.dtype == torch.float32

            g_float16 = torch.mm(d_float32, f_float32)
            # We should be back in fp16
            assert g_float16.dtype == torch.float16

    def test_torch_dynamo_plugin(self):
        with clear_environment():
            prefix = "ACCELERATE_DYNAMO_"
            # nvfuser's dynamo backend name is "nvprims_nvfuser"
            # use "nvfuser" here to cause exception if this test causes os.environ changed permanently
            os.environ[prefix + "BACKEND"] = "aot_ts_nvfuser"
            os.environ[prefix + "MODE"] = "reduce-overhead"

            dynamo_plugin_kwargs = TorchDynamoPlugin().to_kwargs()
            assert dynamo_plugin_kwargs == {"backend": "aot_ts_nvfuser", "mode": "reduce-overhead"}
        assert os.environ.get(prefix + "BACKEND") != "aot_ts_nvfuser"


def main():
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])

    # Skip this test due to TorchXLA not using torch.nn.parallel.DistributedDataParallel for model wrapping.
    if accelerator.distributed_type == DistributedType.XLA:
        return

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


if __name__ == "__main__":
    main()
