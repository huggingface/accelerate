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
from accelerate.test_utils import require_cpu, require_cuda, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase


class AMPOptimizerTests:
    def check_amp_overflow(self, device):
        Accelerator(cpu=device == "cpu")
        for fused in (False, True):
            with self.subTest(fused=fused):
                parameter = torch.nn.Parameter(torch.ones(4, device=device))
                raw_optimizer = torch.optim.AdamW([parameter], lr=0.1, fused=fused)
                scaler = torch.amp.GradScaler(device, growth_interval=2)
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

                    with patch.object(scaler, "get_scale", side_effect=AssertionError("unexpected host scale read")):
                        optimizer.step()
                    scheduler.step()

                    self.assertIs(optimizer.step_was_skipped, overflow)
                    self.assertIs(optimizer.step_was_skipped, overflow)
                    self.assertEqual(torch.equal(parameter, before), overflow)
                    self.assertEqual(float(raw_optimizer.state[parameter]["step"]), step_before + (not overflow))
                    self.assertEqual(scheduler.scheduler.last_epoch, scheduler_before + (not overflow))

    def check_deferred_amp_status(self, device):
        Accelerator(cpu=device == "cpu")
        parameter = torch.nn.Parameter(torch.ones(4, device=device))
        scaler = torch.amp.GradScaler(device)
        optimizer = AcceleratedOptimizer(torch.optim.AdamW([parameter], fused=True), scaler=scaler)
        scaler.scale(parameter.sum()).backward()
        parameter.grad[0] = torch.inf

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as step_profile:
            optimizer.step()
        self.assertNotIn("aten::_local_scalar_dense", [event.key for event in step_profile.key_averages()])

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as read_profile:
            self.assertIs(optimizer.step_was_skipped, True)
            self.assertIs(optimizer.step_was_skipped, True)
        scalar_reads = sum(
            event.count for event in read_profile.key_averages() if event.key == "aten::_local_scalar_dense"
        )
        self.assertEqual(scalar_reads, 1)


@require_cpu
class CPUOptimizerTester(AMPOptimizerTests, AccelerateTestCase):
    def test_amp_overflow_keeps_optimizer_and_scheduler_in_sync(self):
        self.check_amp_overflow("cpu")

    def test_amp_status_is_materialized_once_on_demand(self):
        self.check_deferred_amp_status("cpu")

    def test_amp_status_survives_shared_scaler_updates_and_pickling(self):
        Accelerator(cpu=True)
        scaler = torch.amp.GradScaler("cpu")
        optimizers = []
        for overflow in (True, False):
            parameter = torch.nn.Parameter(torch.ones(4))
            optimizer = AcceleratedOptimizer(torch.optim.AdamW([parameter], fused=True), scaler=scaler)
            scaler.scale(parameter.sum()).backward()
            if overflow:
                parameter.grad[0] = torch.inf
            optimizer.step()
            optimizers.append(optimizer)

        restored = pickle.loads(pickle.dumps(optimizers[0]))
        self.assertIs(restored.step_was_skipped, True)
        self.assertIs(optimizers[0].step_was_skipped, True)
        self.assertIs(optimizers[1].step_was_skipped, False)

    def test_disabled_scaler_does_not_reuse_overflow_status(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.ones(4))
        optimizer = AcceleratedOptimizer(
            torch.optim.AdamW([parameter], fused=True), scaler=torch.amp.GradScaler("cpu", enabled=False)
        )
        parameter.sum().backward()
        optimizer.step()
        self.assertIs(optimizer.step_was_skipped, False)

    def test_accelerated_optimizer_pickling(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator()
        optimizer = accelerator.prepare(optimizer)
        try:
            pickle.loads(pickle.dumps(optimizer))
        except Exception as e:
            self.fail(f"Accelerated optimizer pickling failed with {e}")


@require_cuda
class CUDAAMPOptimizerTester(AMPOptimizerTests, AccelerateTestCase):
    def test_amp_overflow_keeps_optimizer_and_scheduler_in_sync(self):
        self.check_amp_overflow("cuda")

    def test_amp_status_is_materialized_once_on_demand(self):
        self.check_deferred_amp_status("cuda")


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
