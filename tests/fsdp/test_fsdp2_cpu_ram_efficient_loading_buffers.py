# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Test that FSDP2 + cpu_ram_efficient_loading=True correctly loads models that have buffers.
Regression test for https://github.com/huggingface/accelerate/issues/3898
"""

import argparse
import unittest
from typing import Any, Callable

import torch
from torch import distributed as dist
from torch import nn

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.test_utils import (
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_multi_device,
)
from accelerate.test_utils.testing import (
    require_fsdp2,
    require_non_cpu,
    require_non_torch_xla,
    run_first,
    slow,
    torch_device,
)
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils.imports import is_hpu_available


def manage_process_group(func: Callable[..., Any]) -> Callable[..., Any]:
    """Manage the creation and destruction of the distributed process group for the wrapped function."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
        if torch_device == "hpu" and is_hpu_available(init_hccl=True):
            dist.init_process_group(backend="hccl", world_size=torch_accelerator_module.device_count())
        else:
            dist.init_process_group(world_size=torch_accelerator_module.device_count())
        try:
            return func(*args, **kwargs)
        finally:
            dist.destroy_process_group()

    return wrapped


class TinyModelWithBuffer(nn.Module):
    """Minimal model with both parameters and a buffer to exercise buffer loading in fsdp2_load_full_state_dict."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("my_buffer", torch.ones(2, 2) * 42.0)

    def forward(self, x):
        return self.linear(x)


@manage_process_group
def _run_fsdp2_cpu_ram_efficient_loading_buffers():
    torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
    torch_accelerator_module.set_device(device := torch.device(dist.get_rank()))

    plugin = FullyShardedDataParallelPlugin(fsdp_version=2, cpu_ram_efficient_loading=True)
    parallelism_config = ParallelismConfig(dp_shard_size=dist.get_world_size())
    accelerator = Accelerator(fsdp_plugin=plugin, parallelism_config=parallelism_config)

    model = TinyModelWithBuffer()
    expected_buffer = model.my_buffer.detach().clone()

    model = accelerator.prepare(model)

    # Buffer should have been loaded correctly on all ranks (broadcast from rank 0).
    # Use named_buffers() so we get the buffer regardless of FSDP wrapping (e.g. _fsdp_wrapped_module).
    loaded_buffer = next((b for n, b in model.named_buffers() if n == "my_buffer"), None)
    assert loaded_buffer is not None, "Buffer 'my_buffer' not found after prepare"
    torch.testing.assert_close(
        loaded_buffer.cpu(), expected_buffer.cpu(), msg="Buffer mismatch after FSDP2 prepare with cpu_ram_efficient_loading"
    )


class TestFSDP2CpuRamEfficientLoadingBuffersUnit(unittest.TestCase):
    """Unit tests that run without multi-GPU: validate test model and that the test script is runnable."""

    def test_tiny_model_has_buffer(self):
        """TinyModelWithBuffer must have a buffer so fsdp2_load_full_state_dict sees both DTensor and Tensor."""
        model = TinyModelWithBuffer()
        self.assertIn("my_buffer", dict(model.named_buffers()))
        expected = torch.ones(2, 2) * 42.0
        torch.testing.assert_close(model.my_buffer, expected)

    def test_script_accepts_fsdp2_buffers_arg(self):
        """The test script must be invokable with --fsdp2_buffers (used by torchrun)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--fsdp2_buffers", action="store_true")
        args = parser.parse_args(["--fsdp2_buffers"])
        self.assertTrue(args.fsdp2_buffers)


@require_fsdp2
@run_first
@require_non_cpu
@require_non_torch_xla
@require_multi_device
@slow
class TestFSDP2CpuRamEfficientLoadingBuffers(unittest.TestCase):
    """Regression test for issue #3898: FSDP2 + cpu_ram_efficient_loading with models that have buffers."""

    def setUp(self):
        self.torch_accelerator_module = getattr(torch, torch_device, torch.cuda)

    def test_fsdp2_cpu_ram_efficient_loading_buffers(self):
        """Runs the actual FSDP2 + cpu_ram_efficient_loading path with a model that has a buffer."""
        execute_subprocess_async(
            cmd=[
                "torchrun",
                f"--nproc_per_node={self.torch_accelerator_module.device_count()}",
                f"--master_port={get_torch_dist_unique_port()}",
                __file__,
                "--fsdp2_buffers",
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp2_buffers", action="store_true", help="Run FSDP2 cpu_ram_efficient_loading buffers test")
    args = parser.parse_args()

    if args.fsdp2_buffers:
        _run_fsdp2_cpu_ram_efficient_loading_buffers()
    else:
        raise ValueError("Pass --fsdp2_buffers to run the test")
