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
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.test_utils import require_cpu, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase


class ScheduleFreeLikeOptimizer(torch.optim.SGD):
    """Stands in for a `schedule_free` optimizer, which tracks a train/eval mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = None

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


class DeepSpeedLikeOptimizerWrapper:
    """
    Stands in for a DeepSpeed optimizer, which wraps the user optimizer one level deeper. It deliberately does not
    expose `train`/`eval` itself, which is why `AcceleratedOptimizer` has to reach through to the wrapped optimizer.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


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

    def test_accelerated_optimizer_train_eval(self):
        Accelerator()
        model = torch.nn.Linear(10, 10)
        inner = ScheduleFreeLikeOptimizer(model.parameters(), 0.1)
        optimizer = AcceleratedOptimizer(inner)

        optimizer.train()
        assert inner.mode == "train"
        optimizer.eval()
        assert inner.mode == "eval"

    def test_accelerated_optimizer_train_eval_with_wrapped_optimizer(self):
        Accelerator()
        model = torch.nn.Linear(10, 10)
        inner = ScheduleFreeLikeOptimizer(model.parameters(), 0.1)
        optimizer = AcceleratedOptimizer(DeepSpeedLikeOptimizerWrapper(inner))

        optimizer.train()
        assert inner.mode == "train"
        optimizer.eval()
        assert inner.mode == "eval"

    def test_accelerated_optimizer_train_eval_without_mode_support(self):
        Accelerator()
        model = torch.nn.Linear(10, 10)
        inner = torch.optim.SGD(model.parameters(), 0.1)
        optimizer = AcceleratedOptimizer(inner)

        optimizer.train()
        optimizer.eval()


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
