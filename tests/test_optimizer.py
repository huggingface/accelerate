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
from unittest.mock import patch

import torch

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.test_utils import require_cpu, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase, require_torch_min_version


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


@require_torch_min_version(version="2.3.0")
class OptimizerExceptionTester(AccelerateTestCase):
    def test_scaler_exception_restores_optimizer_step(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        original_step = optimizer.step
        scaler = torch.amp.GradScaler("cpu")
        accelerated_optimizer = AcceleratedOptimizer(optimizer, scaler=scaler)
        with self.assertRaisesRegex(AssertionError, "Attempted step but _scale is None"):
            accelerated_optimizer.step()

        self.assertEqual(optimizer.step, original_step)
        self.assertFalse(accelerated_optimizer._accelerate_step_called)
        scaler.scale(parameter.sum()).backward()
        accelerated_optimizer.step()
        torch.testing.assert_close(parameter, torch.tensor([0.9]))

    def test_scaler_update_exception_restores_optimizer_step(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        original_step = optimizer.step
        scaler = torch.amp.GradScaler("cpu")
        accelerated_optimizer = AcceleratedOptimizer(optimizer, scaler=scaler)
        scaler.scale(parameter.sum()).backward()
        error = RuntimeError("scaler update failed")
        with patch.object(scaler, "update", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                accelerated_optimizer.step()

        self.assertIs(caught.exception, error)
        self.assertEqual(optimizer.step, original_step)
        self.assertFalse(accelerated_optimizer._accelerate_step_called)
        torch.testing.assert_close(parameter, torch.tensor([0.9]))

    def test_optimizer_exception_does_not_hide_next_overflow(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        original_step = optimizer.step
        scaler = torch.amp.GradScaler("cpu")
        accelerated_optimizer = AcceleratedOptimizer(optimizer, scaler=scaler)
        accelerated_scheduler = AcceleratedScheduler(scheduler, accelerated_optimizer)

        error = RuntimeError("optimizer hook failed")

        def failing_hook(optimizer, args, kwargs):
            raise error

        handle = optimizer.register_step_pre_hook(failing_hook)
        scaler.scale(parameter.sum()).backward()
        with self.assertRaises(RuntimeError) as caught:
            accelerated_optimizer.step()
        self.assertIs(caught.exception, error)
        self.assertEqual(optimizer.step, original_step)
        self.assertFalse(accelerated_optimizer._accelerate_step_called)
        handle.remove()

        # Finish the failed scaler iteration before recovering with another batch.
        scaler.update()
        accelerated_optimizer.zero_grad()
        scaler.scale(parameter.sum()).backward()
        parameter.grad.fill_(float("inf"))
        accelerated_optimizer.step()
        accelerated_scheduler.step()

        torch.testing.assert_close(parameter, torch.tensor([1.0]))
        self.assertTrue(accelerated_optimizer.step_was_skipped)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(optimizer.step, original_step)
        self.assertFalse(accelerated_optimizer._accelerate_step_called)

        accelerated_optimizer.zero_grad()
        scaler.scale(parameter.sum()).backward()
        accelerated_optimizer.step()
        accelerated_scheduler.step()
        torch.testing.assert_close(parameter, torch.tensor([0.9]))
        self.assertFalse(accelerated_optimizer.step_was_skipped)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)


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
