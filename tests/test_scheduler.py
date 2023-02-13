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

import torch

from accelerate import Accelerator, debug_launcher
from accelerate.state import AcceleratorState
from accelerate.test_utils import require_cpu


def one_cycle_test(num_processes=2, step_scheduler_with_optimizer=True, split_batches=False):
    accelerator = Accelerator(step_scheduler_with_optimizer=step_scheduler_with_optimizer, split_batches=split_batches)
    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Optimizer has stepped
    scheduler.step()
    if step_scheduler_with_optimizer or (num_processes == 1):
        assert (
            scheduler.scheduler.last_epoch == num_processes
        ), f"Last Epoch ({scheduler.scheduler.last_epoch}) != Num Processes ({num_processes})"
    else:
        assert (
            scheduler.scheduler.last_epoch != num_processes
        ), f"Last Epoch ({scheduler.scheduler.last_epoch}) == Num Processes ({num_processes})"


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
    assert (
        scheduler.get_last_lr()[0] == expected_lr
    ), f"Wrong lr found at first step, expected {expected_lr}, got {scheduler.get_last_lr()[0]}"

    # Optimizer has not stepped
    optimizer._is_overflow = True
    scheduler.step()
    if not step_scheduler_with_optimizer:
        expected_lr = 1 - 2 / 10
    assert (
        scheduler.get_last_lr()[0] == expected_lr
    ), f"Wrong lr found at second step, expected {expected_lr}, got {scheduler.get_last_lr()[0]}"


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
