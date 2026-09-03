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

import torch
from torch.distributed.tensor import DTensor

from accelerate import Accelerator


model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)

for parameter, state in optimizer.state.items():
    assert isinstance(parameter, DTensor)
    assert isinstance(state["sum"], DTensor)
    assert state["sum"].device_mesh == parameter.device_mesh
    assert state["sum"].placements == parameter.placements
    assert state["sum"].device == parameter.device

inputs = torch.randn(8, 16, device=accelerator.device)
loss = model(inputs).sum()
accelerator.backward(loss)
optimizer.step()
optimizer.state_dict()
