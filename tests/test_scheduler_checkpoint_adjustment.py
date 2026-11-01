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

import copy

import pytest
import torch

from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import GradientAccumulationPlugin


@pytest.mark.parametrize("adjust", [False, True])
def test_one_cycle_accumulation_checkpoint_roundtrip(adjust):
    def make():
        GradientState._reset_state()
        AcceleratorState._reset_state(True)
        accelerator = Accelerator(
            cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=2, adjust_scheduler=adjust)
        )
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=4)
        return accelerator, *accelerator.prepare(model, optimizer, scheduler)

    def advance(parts, count):
        accelerator, model, optimizer, scheduler = parts
        rates = []
        for _ in range(count):
            with accelerator.accumulate(model):
                accelerator.backward(model(torch.ones(1, 2)).sum())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                rates.append(scheduler.get_last_lr())
        return rates

    try:
        parts = make()
        advance(parts, 4)
        optimizer_state = copy.deepcopy(parts[2].state_dict())
        scheduler_state = copy.deepcopy(parts[3].state_dict())
        expected = advance(parts, 4)
        resumed = make()
        resumed[3].load_state_dict(scheduler_state)
        resumed[2].load_state_dict(optimizer_state)
        assert advance(resumed, 4) == expected
        assert resumed[3].scheduler.last_epoch == 4
        assert resumed[3].scheduler._step_count == (9 if adjust else 5)
    finally:
        GradientState._reset_state()
        AcceleratorState._reset_state(True)
