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
Checks that `Accelerator.clip_grad_value_` clips every rank's shard of an FSDP2 gradient, after unscaling it.

The gradient of `model(x).sum()` is `x` on every row of the weight, whatever the weight, so the clipped gradient is known
exactly. Under fp16 the loss scale is 8, so clipping before unscaling would turn 0.5 (scaled to 4) into 1.
"""

import torch

from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs


def main():
    accelerator = Accelerator(kwargs_handlers=[GradScalerKwargs(init_scale=8.0)])
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model, _ = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    accelerator.backward(model(torch.tensor([[3.0, 0.5, -3.0, -0.5]], device=accelerator.device)).sum())
    accelerator.clip_grad_value_(model.parameters(), 1.0)

    grad = model[0].weight.grad
    assert grad.to_local().shape == (2, 4), f"rank {accelerator.process_index} holds {tuple(grad.to_local().shape)}"
    assert grad.full_tensor().tolist() == [[1.0, 0.5, -1.0, -0.5]] * 4, grad.full_tensor().tolist()
    accelerator.end_training()


if __name__ == "__main__":
    main()
