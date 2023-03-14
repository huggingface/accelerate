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

# We ignore warnings about stepping the scheduler since we step it ourselves during gradient accumulation

import warnings

from .state import AcceleratorState, GradientState


warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class AcceleratedScheduler:
    """
    A wrapper around a learning rate scheduler that will only step when the optimizer(s) have a training step. Useful
    to avoid making a scheduler step too fast when gradients went overflow and there was no training step (in mixed
    precision training)

    When performing gradient accumulation scheduler lengths should not be changed accordingly, Accelerate will always
    step the scheduler to account for it.

    Args:
        scheduler (`torch.optim.lr_scheduler._LRScheduler`):
            The scheduler to wrap.
        optimizers (one or a list of `torch.optim.Optimizer`):
            The optimizers used.
        step_with_optimizer (`bool`, *optional*, defaults to `True`):
            Whether or not the scheduler should be stepped at each optimizer step.
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the dataloaders split one batch across the different processes (so batch size is the same
            regardless of the number of processes) or create batches on each process (so batch size is the original
            batch size multiplied by the number of processes).
    """

    def __init__(self, scheduler, optimizers, step_with_optimizer: bool = True, split_batches: bool = False):
        self.scheduler = scheduler
        self.optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        self.split_batches = split_batches
        self.step_with_optimizer = step_with_optimizer
        self.gradient_state = GradientState()

    def step(self, *args, **kwargs):
        if not self.step_with_optimizer:
            # No link between scheduler and optimizer -> just step
            self.scheduler.step(*args, **kwargs)
            return

        # Otherwise, first make sure the optimizer was stepped.
        if not self.gradient_state.sync_gradients:
            if self.gradient_state.adjust_scheduler:
                self.scheduler._step_count += 1
            return

        for opt in self.optimizers:
            if opt.step_was_skipped:
                return
        if self.split_batches:
            # Split batches -> the training dataloader batch size is not changed so one step per training step
            self.scheduler.step(*args, **kwargs)
        else:
            # Otherwise the training dataloader batch size was multiplied by `num_processes`, so we need to do
            # num_processes steps per training step
            num_processes = AcceleratorState().num_processes
            for _ in range(num_processes):
                # Special case when using OneCycle and `drop_last` was not used
                if hasattr(self.scheduler, "total_steps"):
                    if self.scheduler._step_count <= self.scheduler.total_steps:
                        self.scheduler.step(*args, **kwargs)
                else:
                    self.scheduler.step(*args, **kwargs)

    # Passthroughs
    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        return self.scheduler.get_lr()

    def print_lr(self, *args, **kwargs):
        return self.scheduler.print_lr(*args, **kwargs)
