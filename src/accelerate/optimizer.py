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

import inspect
import warnings

import torch

from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_torch_version, is_tpu_available


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def move_to_device(state, device):
    if isinstance(state, (list, tuple)):
        return honor_type(state, (move_to_device(t, device) for t in state))
    elif isinstance(state, dict):
        return type(state)({k: move_to_device(v, device) for k, v in state.items()})
    elif isinstance(state, torch.Tensor):
        return state.to(device)
    return state


class AcceleratedOptimizer(torch.optim.Optimizer):
    """
    Internal wrapper around a torch optimizer.

    Conditionally will perform `step` and `zero_grad` if gradients should be synchronized when performing gradient
    accumulation.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the optimizer should handle device placement. If so, it will place the state dictionary of
            `optimizer` on the right device.
        scaler (`torch.cuda.amp.grad_scaler.GradScaler`, *optional*):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.accelerator_state = AcceleratorState()
        self.gradient_state = GradientState()
        self.device_placement = device_placement
        self._is_overflow = False

        # Handle device placement
        if device_placement:
            state_dict = self.optimizer.state_dict()
            if self.accelerator_state.distributed_type == DistributedType.TPU:
                xm.send_cpu_data_to_device(state_dict, self.accelerator_state.device)
            else:
                state_dict = move_to_device(state_dict, self.accelerator_state.device)
            self.optimizer.load_state_dict(state_dict)

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        if self.accelerator_state.distributed_type == DistributedType.TPU and self.device_placement:
            xm.send_cpu_data_to_device(state_dict, self.accelerator_state.device)
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if self.gradient_state.sync_gradients:
            if is_torch_version("<", "1.7.0"):
                if set_to_none is not None:
                    raise ValueError(
                        "`set_to_none` for Optimizer.zero_grad` was introduced in PyTorch 1.7.0 and can't be used for "
                        f"earlier versions (found version {torch.__version__})."
                    )
                self.optimizer.zero_grad()
            else:
                accept_arg = "set_to_none" in inspect.signature(self.optimizer.zero_grad).parameters
                if accept_arg:
                    if set_to_none is None:
                        set_to_none = False
                    self.optimizer.zero_grad(set_to_none=set_to_none)
                else:
                    if set_to_none is not None:
                        raise ValueError("`set_to_none` for Optimizer.zero_grad` is not supported by this optimizer.")
                    self.optimizer.zero_grad()

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            if self.accelerator_state.distributed_type == DistributedType.TPU:
                optimizer_args = {"closure": closure} if closure is not None else {}
                xm.optimizer_step(self.optimizer, optimizer_args=optimizer_args)
            elif self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer, closure)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                # If we reduced the loss scale, it means the optimizer step was skipped because of gradient overflow.
                self._is_overflow = scale_after < scale_before
            else:
                self.optimizer.step(closure)

    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]

    @property
    def is_overflow(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        warnings.warn(
            "The `is_overflow` property is deprecated and will be removed in version 1.0 of Accelerate use "
            "`optimizer.step_was_skipped` instead.",
            FutureWarning,
        )
        return self._is_overflow

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was skipped."""
        return self._is_overflow
