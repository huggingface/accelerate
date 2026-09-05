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

import pickle

import torch

from accelerate import Accelerator
from accelerate.test_utils import require_cpu, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase


@require_cpu
class CPUOptimizerTester(AccelerateTestCase):
    def test_accelerated_optimizer_pickling(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator()
        optimizer = accelerator.prepare(optimizer)
        try:
            pickle.loads(pickle.dumps(optimizer))
        except Exception as e:
            self.fail(f"Accelerated optimizer pickling failed with {e}")

    def test_switch_parameters_updates_optimizer_state(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adagrad(model.parameters(), 0.1)
        accelerator = Accelerator()
        optimizer = accelerator.prepare(optimizer)

        old_parameters = list(optimizer.param_groups[0]["params"])
        old_states = [optimizer.state[parameter] for parameter in old_parameters]
        new_parameters = [torch.nn.Parameter(torch.empty_like(parameter)) for parameter in old_parameters]
        optimizer._switch_parameters(dict(zip(old_parameters, new_parameters)))

        assert all(actual is expected for actual, expected in zip(optimizer.param_groups[0]["params"], new_parameters))
        assert set(optimizer.state) == set(new_parameters)
        assert all(optimizer.state[parameter] is state for parameter, state in zip(new_parameters, old_states))
        optimizer.state_dict()


@require_fp16
@require_non_cpu
class OptimizerTester(AccelerateTestCase):
    def test_accelerated_optimizer_step_was_skipped(self):
        model = torch.nn.Linear(5, 5)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator(mixed_precision="fp16")
        model, optimizer = accelerator.prepare(model, optimizer)

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            # Fake the gradients, as if there's no overflow
            p.grad.fill_(0.01)

        optimizer.step()
        assert optimizer.step_was_skipped is False

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            p.grad.fill_(0.01)
        # Manually set the gradients to be NaN, as if there's an overflow
        p.grad[0] = torch.tensor(float("nan"))

        optimizer.step()
        assert optimizer.step_was_skipped is True

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            p.grad.fill_(0.01)
        # Manually set the gradients to be NaN, as if there's an overflow
        p.grad[0] = torch.tensor(float("nan"))

        optimizer.step()
        assert optimizer.step_was_skipped is True

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            # Fake the gradients, as if there's no overflow
            p.grad.fill_(0.01)

        optimizer.step()
        assert optimizer.step_was_skipped is False
