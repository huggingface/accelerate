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

from accelerate.scheduler import AcceleratedScheduler

from ..optimizer import AcceleratedOptimizer


class DeepSpeedEngineWrapper:
    def __init__(self, engine):
        self.engine = engine

    def backward(self, loss):
        # runs backpropagation and handles mixed precision
        self.engine.backward(loss)

        # gradient accumulation check
        # gradient clipping
        # optimizer step
        # zero grad
        # checking overflow
        # lr_scheduler step
        self.engine.step()


class DeepSpeedOptimizerWrapper(AcceleratedOptimizer):
    """
    Internal wrapper around a deepspeed optimizer.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)

    def zero_grad(self, set_to_none=None):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, it's implementation is not needed

    def step(self):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, it's implementation is not needed

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        return self.optimizer.overflow


class DeepSpeedSchedulerWrapper(AcceleratedScheduler):
    """
    Internal wrapper around a deepspeed scheduler.

    Args:
        scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler to wrap.
        optimizers (one or a list of `torch.optim.Optimizer`):
    """

    def __init__(self, scheduler, optimizers):
        super().__init__(scheduler, optimizers)

    def step(self):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, it's implementation is not needed


class DummyOptim:
    """
    Dummy optimizer presents model parameters or param groups, this is primarily used to follow conventional training
    loop when optimizer config is specified in the deepspeed config file.

    Args:
        params (list of `torch.Tensor`):
            List of model parameters.
        param_groups (list of dict):
            List of dictionaries with parameters and their groups.
    """

    def __init__(self, params=None, param_groups=None):
        self.param_groups = []
        if params is None and param_groups is None:
            raise ValueError("Either params or param_groups should be specified")
        if param_groups is None:
            self.param_groups.append({"params": params})
        else:
            self.param_groups = param_groups


class DummyScheduler:
    """
    Dummy scheduler presents model parameters or param groups, this is primarily used to follow conventional training
    loop when scheduler config is specified in the deepspeed config file.
    """

    def __init__(self):
        pass
