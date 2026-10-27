# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from unittest import skipUnless
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate.accelerator import Accelerator
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import is_torch_version


if dist.is_available() and is_torch_version(">=", "2.6"):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor


@contextmanager
def _distributed_context(rank, directory):
    environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "4"}
    with patch.dict(os.environ, environment), patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=60),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def _gradient_clipping_worker(rank, directory):
    with _distributed_context(rank, directory):
        mesh = init_device_mesh("cpu", (4,))
        # Same seed on every rank, so all ranks build the same reference gradients.
        torch.manual_seed(0)
        for norm_type in (0.0, 2.0, float("inf")):
            for distributed in ((False, False), (True, True), (False, True)):
                for max_norm in (1.0, 10000.0):
                    parameters, reference = [], []
                    for is_distributed in distributed:
                        gradient = torch.randn(8, 8)
                        expected = torch.nn.Parameter(torch.zeros_like(gradient))
                        expected.grad = gradient.clone()
                        reference.append(expected)
                        if is_distributed:
                            gradient = distribute_tensor(gradient, mesh, [Shard(0)])
                        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
                        parameter.grad = gradient
                        parameters.append(parameter)
                    expected_norm = torch.nn.utils.clip_grad_norm_(
                        reference, max_norm, foreach=True, norm_type=norm_type
                    )
                    actual_norm = Accelerator._clip_grad_norm_dtensor_aware(
                        None, parameters, max_norm, foreach=True, norm_type=norm_type
                    )
                    torch.testing.assert_close(
                        actual_norm.full_tensor() if isinstance(actual_norm, DTensor) else actual_norm, expected_norm
                    )
                    for parameter, expected in zip(parameters, reference):
                        grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
                        torch.testing.assert_close(grad, expected.grad)


class DistributedUtilsTester(AccelerateTestCase):
    @skipUnless(
        dist.is_available() and dist.is_gloo_available() and is_torch_version(">=", "2.6"),
        "Requires distributed Gloo and PyTorch >= 2.6",
    )
    def test_clip_grad_norm_dtensor_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_gradient_clipping_worker, args=(directory,), nprocs=4, join=True)
