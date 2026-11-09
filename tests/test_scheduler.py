# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from functools import partial
from itertools import product

import torch
from parameterized import parameterized

from accelerate import Accelerator, debug_launcher
from accelerate.state import AcceleratorState, GradientState
from accelerate.test_utils import require_cpu, require_huggingface_suite
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import GradientAccumulationPlugin


def one_cycle_test(num_processes=2, step_scheduler_with_optimizer=True, split_batches=False):
    accelerator = Accelerator(step_scheduler_with_optimizer=step_scheduler_with_optimizer, split_batches=split_batches)
    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Optimizer has stepped
    scheduler.step()
    if step_scheduler_with_optimizer or (num_processes == 1):
        assert scheduler.scheduler.last_epoch == num_processes, (
            f"Last Epoch ({scheduler.scheduler.last_epoch}) != Num Processes ({num_processes})"
        )
    else:
        assert scheduler.scheduler.last_epoch != num_processes, (
            f"Last Epoch ({scheduler.scheduler.last_epoch}) == Num Processes ({num_processes})"
        )


def lambda_test(num_processes=2, step_scheduler_with_optimizer=True, split_batches=False):
    accelerator = Accelerator(step_scheduler_with_optimizer=step_scheduler_with_optimizer, split_batches=split_batches)
    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: 1 - n / 10)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Optimizer has stepped
    optimizer._is_overflow = False
    scheduler.step()
    expected_lr = 1 - (num_processes if (step_scheduler_with_optimizer and not split_batches) else 1) / 10
    assert scheduler.get_last_lr()[0] == expected_lr, (
        f"Wrong lr found at first step, expected {expected_lr}, got {scheduler.get_last_lr()[0]}"
    )

    # Optimizer has not stepped
    optimizer._is_overflow = True
    scheduler.step()
    if not step_scheduler_with_optimizer:
        expected_lr = 1 - 2 / 10
    assert scheduler.get_last_lr()[0] == expected_lr, (
        f"Wrong lr found at second step, expected {expected_lr}, got {scheduler.get_last_lr()[0]}"
    )


def accumulation_test(num_processes: int = 2):
    """
    With this test, an observed batch size of 64 should result in negligible
    differences in the scheduler after going through the correct number of steps.

    Uses single, two, and four steps to test.
    """
    from transformers import get_linear_schedule_with_warmup

    steps = [1, 2, 4]
    for num_steps in steps:
        plugin = GradientAccumulationPlugin(num_steps=num_steps, adjust_scheduler=num_steps > 1)
        accelerator = Accelerator(gradient_accumulation_plugin=plugin)
        model = torch.nn.Linear(2, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=10.0)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=20)

        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        for i in range(10 * num_steps):
            with accelerator.accumulate(model):
                optimizer.step()
                scheduler.step()

            if i == (10 * num_steps - 2):
                assert scheduler.get_last_lr()[0] != 0, (
                    f"Wrong lr found at second-to-last step, expected non-zero, got {scheduler.get_last_lr()[0]}. num_steps: {num_steps}"
                )
        assert scheduler.get_last_lr()[0] == 0, (
            f"Wrong lr found at last step, expected 0, got {scheduler.get_last_lr()[0]}"
        )
        GradientState._reset_state()


@require_cpu
class SchedulerTester(unittest.TestCase):
    def test_lambda_scheduler_steps_with_optimizer_single_process(self):
        debug_launcher(partial(lambda_test, num_processes=1), num_processes=1)
        debug_launcher(partial(lambda_test, num_processes=1, split_batches=True), num_processes=1)

    def test_one_cycle_scheduler_steps_with_optimizer_single_process(self):
        debug_launcher(partial(one_cycle_test, num_processes=1), num_processes=1)
        debug_launcher(partial(one_cycle_test, num_processes=1, split_batches=True), num_processes=1)

    def test_lambda_scheduler_not_step_with_optimizer_single_process(self):
        debug_launcher(partial(lambda_test, num_processes=1, step_scheduler_with_optimizer=False), num_processes=1)

    def test_one_cycle_scheduler_not_step_with_optimizer_single_process(self):
        debug_launcher(partial(one_cycle_test, num_processes=1, step_scheduler_with_optimizer=False), num_processes=1)

    def test_lambda_scheduler_steps_with_optimizer_multiprocess(self):
        AcceleratorState._reset_state(True)
        debug_launcher(lambda_test)
        debug_launcher(partial(lambda_test, num_processes=1, split_batches=True), num_processes=1)

    def test_one_cycle_scheduler_steps_with_optimizer_multiprocess(self):
        AcceleratorState._reset_state(True)
        debug_launcher(one_cycle_test)
        debug_launcher(partial(one_cycle_test, num_processes=1, split_batches=True), num_processes=1)

    def test_lambda_scheduler_not_step_with_optimizer_multiprocess(self):
        AcceleratorState._reset_state(True)
        debug_launcher(partial(lambda_test, step_scheduler_with_optimizer=False))

    def test_one_cycle_scheduler_not_step_with_optimizer_multiprocess(self):
        AcceleratorState._reset_state(True)
        debug_launcher(partial(one_cycle_test, step_scheduler_with_optimizer=False))

    @require_huggingface_suite
    def test_accumulation(self):
        AcceleratorState._reset_state(True)
        debug_launcher(partial(accumulation_test, num_processes=1))
        debug_launcher(accumulation_test)


class SchedulerOptimizerAssociationTester(AccelerateTestCase):
    @parameterized.expand(product([False, True], ["prepare", "prepare_scheduler"], [0, 1], [False, True]))
    def test_scheduler_tracks_only_its_optimizer(
        self, scheduler_after_prepare, prepare_method, optimizer_index, overflow_own_optimizer
    ):
        accelerator = Accelerator(cpu=True)
        # Use a real CPU scaler to exercise overflow without requiring a GPU.
        accelerator.scaler = torch.amp.GradScaler("cpu")
        parameters = [torch.nn.Parameter(torch.tensor([1.0])) for _ in range(2)]
        optimizers = [torch.optim.SGD([p], lr=0.1) for p in parameters]
        if not scheduler_after_prepare:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizers[optimizer_index], step_size=1, gamma=0.1)
        optimizers = accelerator.prepare(*optimizers)
        if scheduler_after_prepare:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizers[optimizer_index], step_size=1, gamma=0.1)
        scheduler = getattr(accelerator, prepare_method)(scheduler)

        overflow_index = optimizer_index if overflow_own_optimizer else 1 - optimizer_index
        for index, (parameter, optimizer) in enumerate(zip(parameters, optimizers)):
            loss = parameter.sum()
            if index == overflow_index:
                loss = loss * float("inf")
            accelerator.scaler.scale(loss).backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        self.assertTrue(optimizers[overflow_index].step_was_skipped)
        self.assertFalse(optimizers[1 - overflow_index].step_was_skipped)
        expected_lr = 0.1 if overflow_own_optimizer else 0.01
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_lr)
        self.assertAlmostEqual(parameters[optimizer_index].item(), 1.0 if overflow_own_optimizer else 0.9)
        accelerator.scaler.scale(parameters[optimizer_index].sum()).backward()
        optimizers[optimizer_index].step()
        expected_parameter = 0.9 if overflow_own_optimizer else 0.89
        torch.testing.assert_close(parameters[optimizer_index], torch.tensor([expected_parameter]))
