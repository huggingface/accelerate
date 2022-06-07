# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel, require_cpu
from accelerate.utils import set_seed


def step_model(model, input, target, accelerator):
    model.train()
    output = model(input)
    loss = F.mse_loss(output, target.to(output.device))
    accelerator.backward(loss)


@require_cpu
class SyncTest(unittest.TestCase):
    def test_noop_wrapper(self):
        accelerator = Accelerator()
        device = accelerator.device
        set_seed(42)
        model = RegressionModel()
        dset = RegressionDataset()
        dl = DataLoader(dset, batch_size=16)
        ddp_model, dl = accelerator.prepare(deepcopy(model), dl)
        model.to(device)
        ddp_input, ddp_target = next(iter(dl)).values()

        for iteration in range(2):
            input, target = accelerator.gather((ddp_input, ddp_target))
            input = input.to(accelerator.device)
            target = target.to(accelerator.device)
            step_model(model, input, target, accelerator)
            if iteration % 2 == 0:
                # Accumulate grads locally
                with accelerator.no_sync(ddp_model):
                    step_model(ddp_model, ddp_input, ddp_target, accelerator)
            else:
                # Sync grads
                step_model(ddp_model, ddp_input, ddp_target, accelerator)

            for i, j in zip(model.parameters(), ddp_model.parameters()):
                if not i.requires_grad:
                    continue
                assert torch.allclose(i.grad, j.grad), f"{i.grad} != {j.grad}"
