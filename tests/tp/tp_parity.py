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
"""
Numerical parity of a tensor parallel model prepared by Accelerate against a single-device reference.

A small fp32 block (column-parallel `Linear`, row-parallel `Linear`, and a replicated `LayerNorm` with weight and bias,
which `_prepare_tp` has to replicate itself) is tensor-parallelized with `parallelize_module` and prepared with
`ParallelismConfig(tp_size=2)`, optionally with FSDP2 over a second mesh dimension. Every rank also runs the same model
unsharded on one device, on the global batch. The loss, every parameter's gradient and every parameter after one SGD
step must match the reference.

Tolerance: everything runs in fp32 with TF32 disabled, so the only differences come from the order of floating point
accumulation (TP splits the `HIDDEN`-long reductions of the row-parallel layer across ranks and sums the partial
results, FSDP2 averages gradients across data parallel ranks). For a sum of `n` terms, the error of a different
summation order grows like `sqrt(n) * eps` in practice, so we allow `RTOL = 8 * sqrt(HIDDEN) * eps(fp32)` (about
1.5e-5 for `HIDDEN = 256`), relative to the largest magnitude of each reference tensor.

Usage (`accelerate launch --num_processes N tests/tp/tp_parity.py MODE`):
- `tp`: N=2, tensor parallel only
- `tp_fsdp2`: N=4, TP=2 x FSDP2=2
- `tp_fsdp2_cpu_ram_efficient_loading`: N=4, checks that `cpu_ram_efficient_loading=True` is refused with TP
"""

import copy
import math
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed


IN_FEATURES, HIDDEN, BATCH_PER_DP_RANK = 64, 256, 8
LR = 0.1
RTOL = 8 * math.sqrt(HIDDEN) * torch.finfo(torch.float32).eps


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(IN_FEATURES, HIDDEN)
        self.down = torch.nn.Linear(HIDDEN, IN_FEATURES)
        self.norm = torch.nn.LayerNorm(IN_FEATURES)

    def forward(self, x):
        return self.norm(self.down(torch.relu(self.up(x))))


def loss_fn(output, target):
    # A regression loss against a random target, so that no gradient is dominated by cancellation (a loss such as
    # `output.mean()` is nearly constant through `LayerNorm`, and the gradients it leaves are tiny differences of large
    # terms, whose relative error is not a property of the parallel layout)
    return (output - target).pow(2).mean()


def full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def assert_matches(name, actual, expected):
    error = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert error <= RTOL * scale, f"{name}: max abs error {error:.3e} > {RTOL:.3e} * {scale:.3e}"
    return error / scale if scale > 0 else 0.0


def main(mode):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    use_fsdp2 = mode.startswith("tp_fsdp2")
    dp_size = 2 if use_fsdp2 else 1
    kwargs = {"parallelism_config": ParallelismConfig(dp_shard_size=dp_size, tp_size=2)}
    if use_fsdp2:
        kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(
            fsdp_version=2, cpu_ram_efficient_loading=mode.endswith("cpu_ram_efficient_loading")
        )
    accelerator = Accelerator(**kwargs)
    mesh = accelerator.torch_device_mesh

    set_seed(0)
    reference = Block().to(accelerator.device)
    model = copy.deepcopy(reference)
    parallelize_module(model, mesh["tp"], {"up": ColwiseParallel(), "down": RowwiseParallel()})
    model.tp_size = 2
    # Without FSDP2, the replicated `LayerNorm` keeps plain tensor parameters next to the DTensor ones, and the
    # multi-tensor (foreach) optimizer kernels refuse such a mix, so use the per-parameter implementation
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, foreach=use_fsdp2)

    if mode.endswith("cpu_ram_efficient_loading"):
        try:
            accelerator.prepare(model, optimizer)
        except ValueError as error:
            assert "cpu_ram_efficient_loading" in str(error), str(error)
            accelerator.print(f"refused as expected: {error}")
            return
        raise AssertionError("cpu_ram_efficient_loading=True with tensor parallelism was not refused")

    model, optimizer = accelerator.prepare(model, optimizer)

    # The global batch; ranks of the same TP group use the same data parallel slice
    generator = torch.Generator().manual_seed(1234)
    global_batch = torch.randn(dp_size * BATCH_PER_DP_RANK, IN_FEATURES, generator=generator).to(accelerator.device)
    global_target = torch.randn(dp_size * BATCH_PER_DP_RANK, IN_FEATURES, generator=generator).to(accelerator.device)
    dp_rank = mesh["dp_shard"].get_local_rank() if use_fsdp2 else 0
    local_slice = slice(dp_rank * BATCH_PER_DP_RANK, (dp_rank + 1) * BATCH_PER_DP_RANK)

    loss = loss_fn(model(global_batch[local_slice]), global_target[local_slice])
    accelerator.backward(loss)
    # The loss of the global batch is the mean of the equal-sized data parallel slices
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)

    reference_loss = loss_fn(reference(global_batch), global_target)
    reference_loss.backward()

    ratios = {"loss": assert_matches("loss", global_loss, reference_loss.detach())}
    reference_params = dict(reference.named_parameters())
    params = dict(model.named_parameters())
    assert params.keys() == reference_params.keys(), (params.keys(), reference_params.keys())
    for name, param in params.items():
        ratios[f"grad {name}"] = assert_matches(f"grad {name}", full(param.grad), reference_params[name].grad)

    optimizer.step()
    with torch.no_grad():
        for reference_param in reference.parameters():
            reference_param -= LR * reference_param.grad
    for name, param in model.named_parameters():
        ratios[f"param {name}"] = assert_matches(f"param {name}", full(param.detach()), reference_params[name])

    worst = max(ratios, key=ratios.get)
    accelerator.print(
        f"{mode}: loss, {len(params)} gradients and {len(params)} parameters after one SGD step match the "
        f"single-device fp32 reference; largest relative error {ratios[worst]:.2e} ({worst}), tolerance {RTOL:.2e}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
