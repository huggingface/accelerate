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
"""Regression script for FSDP2 + cpu_ram_efficient_loading with persistent buffers (#4382)."""

import torch
from torch import nn

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed


class TinyModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.tensor(1.25))

    def forward(self, x):
        return self.linear(x) * self.scale


def main():
    set_seed(42)
    plugin = FullyShardedDataParallelPlugin(fsdp_version=2, cpu_ram_efficient_loading=True)
    accelerator = Accelerator(fsdp_plugin=plugin)

    model = TinyModelWithBuffer()
    expected = model.scale.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model, optimizer = accelerator.prepare(model, optimizer)

    loaded = dict(model.named_buffers()).get("scale")
    if loaded is None:
        # FSDP wrapping may nest the module; search all buffer names.
        loaded = next((b for n, b in model.named_buffers() if n.endswith("scale")), None)
    assert loaded is not None, "persistent buffer 'scale' missing after prepare"
    torch.testing.assert_close(loaded.detach().cpu(), expected.cpu())
    accelerator.print("ok: persistent buffer survived FSDP2 cpu_ram_efficient_loading")


if __name__ == "__main__":
    main()
