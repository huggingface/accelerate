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
from accelerate.scheduler import AcceleratedScheduler
from accelerate.test_utils import require_cpu, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase


@require_cpu
class CPUOptimizerTester(AccelerateTestCase):
    def test_amp_overflow_keeps_optimizer_and_scheduler_in_sync(self):
        Accelerator(cpu=True)
        for fused in (False, True):
            with self.subTest(fused=fused):
                parameter = torch.nn.Parameter(torch.ones(4))
                raw_optimizer = torch.optim.AdamW([parameter], lr=0.1, fused=fused)
                scaler = torch.amp.GradScaler("cpu", growth_interval=2)
                optimizer = AcceleratedOptimizer(raw_optimizer, scaler=scaler)
                scheduler = AcceleratedScheduler(
                    torch.optim.lr_scheduler.LambdaLR(raw_optimizer, lambda _: 1.0),
                    optimizer,
                    split_batches=True,
                )

                # Include consecutive overflows, recovery, and a successful scale-growth step.
                for overflow in (False, True, True, False, False):
                    optimizer.zero_grad()
                    scaler.scale(parameter.sum()).backward()
                    if overflow:
                        parameter.grad[0] = torch.inf
                    before = parameter.detach().clone()
                    step_before = float(raw_optimizer.state.get(parameter, {}).get("step", 0))
                    scheduler_before = scheduler.scheduler.last_epoch

                    optimizer.step()
                    scheduler.step()

                    self.assertEqual(optimizer.step_was_skipped, overflow)
                    self.assertEqual(torch.equal(parameter, before), overflow)
                    self.assertEqual(float(raw_optimizer.state[parameter]["step"]), step_before + (not overflow))
                    self.assertEqual(scheduler.scheduler.last_epoch, scheduler_before + (not overflow))

    def test_accelerated_optimizer_pickling(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator()
        optimizer = accelerator.prepare(optimizer)
        try:
            pickle.loads(pickle.dumps(optimizer))
        except Exception as e:
            self.fail(f"Accelerated optimizer pickling failed with {e}")


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
