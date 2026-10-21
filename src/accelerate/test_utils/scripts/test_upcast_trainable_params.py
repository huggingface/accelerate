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
Trains a model loaded in bf16 for a few steps under DDP with `mixed_precision="bf16"`, with and without
`AutocastKwargs(upcast_trainable_params=True)`, and compares both against a plain PyTorch reference that keeps fp32
master weights: `DistributedDataParallel` on an fp32 copy of the same bf16 weights, the forward under bf16 autocast and
an fp32 AdamW.

With the option, the run must match the reference. Without it, the optimizer steps on bf16 weights, and at a small
learning rate most of the update is rounded away.
"""

import torch

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import AutocastKwargs, set_seed


NUM_STEPS = 5
LR = 1e-5


def make_model():
    set_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))
    # As if loaded with `torch_dtype=torch.bfloat16`
    return model.to(torch.bfloat16)


def batches(rank, device):
    generator = torch.Generator().manual_seed(1000 + rank)
    return [(torch.randn(16, 64, generator=generator).to(device)) for _ in range(NUM_STEPS)]


def train_with_accelerate(upcast):
    AcceleratorState._reset_state()
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[AutocastKwargs(upcast_trainable_params=upcast)])
    model = make_model()
    initial = [param.detach().float().cpu() for param in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model, optimizer = accelerator.prepare(model, optimizer)
    expected_dtype = torch.float32 if upcast else torch.bfloat16
    assert all(param.dtype == expected_dtype for param in model.parameters())
    for inputs in batches(accelerator.process_index, accelerator.device):
        loss = model(inputs).float().pow(2).mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    for param in model.parameters():
        assert optimizer.state[param]["exp_avg"].dtype == expected_dtype
    final = [param.detach().float().cpu() for param in model.parameters()]
    return accelerator, initial, final


def train_reference(accelerator):
    model = make_model().float().to(accelerator.device)  # fp32 master weights holding the same bf16 values
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[accelerator.local_process_index], output_device=accelerator.local_process_index
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    for inputs in batches(accelerator.process_index, accelerator.device):
        with torch.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16):
            output = model(inputs)
        loss = output.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return [param.detach().float().cpu() for param in model.parameters()]


def update_error(initial, final, reference):
    """Relative L2 error of the total update (final - initial) against the reference update."""
    error = sum(((f - r) ** 2).sum() for f, r in zip(final, reference)) ** 0.5
    norm = sum(((r - i) ** 2).sum() for i, r in zip(initial, reference)) ** 0.5
    return (error / norm).item()


def main():
    accelerator, initial, upcast_final = train_with_accelerate(upcast=True)
    reference_final = train_reference(accelerator)
    upcast_error = update_error(initial, upcast_final, reference_final)
    for final, reference in zip(upcast_final, reference_final):
        # Same weights, data, kernels and fp32 optimizer as the reference
        torch.testing.assert_close(final, reference)

    _, initial_bf16, bf16_final = train_with_accelerate(upcast=False)
    bf16_error = update_error(initial_bf16, bf16_final, reference_final)
    accelerator.print(
        f"relative error of the {NUM_STEPS}-step update against the fp32-master reference: "
        f"upcast_trainable_params=True {upcast_error:.3e}, default (bf16 weights) {bf16_error:.3e}"
    )
    assert upcast_error < 1e-3
    # Without fp32 master weights, most of the small updates are lost
    assert bf16_error > 0.1


if __name__ == "__main__":
    main()
