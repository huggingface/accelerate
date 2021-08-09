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

import torch

from packaging import version

from .state import AcceleratorState, DistributedType, is_tpu_available
from .utils import honor_type


if is_tpu_available():
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

    Args:
        optimizer (:obj:`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        device_placement (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the optimizer should handle device placement. If so, it will place the state dictionary of
            :obj:`optimizer` on the right device.
        scaler (:obj:`torch.cuda.amp.grad_scaler.GradScaler`, `optional`):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.state = AcceleratorState()
        self.device_placement = device_placement
        self._is_overflow = False

        # Handle device placement
        if device_placement:
            state_dict = self.optimizer.state_dict()
            if self.state.distributed_type == DistributedType.TPU:
                xm.send_cpu_data_to_device(state_dict, self.state.device)
            else:
                state_dict = move_to_device(state_dict, self.state.device)
            self.optimizer.load_state_dict(state_dict)

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
        if self.state.distributed_type == DistributedType.TPU and self.device_placement:
            xm.send_cpu_data_to_device(state_dict, self.state.device)
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            if set_to_none is not None:
                raise ValueError(
                    "`set_to_none` for Optimizer.zero_grad` was introduced in PyTorch 1.7.0 and can't be used for "
                    f"earlier versions (found version {torch.__version__})."
                )
            self.optimizer.zero_grad()
        else:
            if set_to_none is None:
                set_to_none = False
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self.state.distributed_type == DistributedType.TPU:
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
        return self._is_overflow
