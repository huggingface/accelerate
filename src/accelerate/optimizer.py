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

from .state import AcceleratorState, DistributedType, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


def move_to_device(state, device):
    if isinstance(state, (list, tuple)):
        return type(state)(move_to_device(t, device) for t in state)
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

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        if self.state.distributed_type == DistributedType.TPU and self.device_placement:
            xm.send_cpu_data_to_device(state_dict, self.state.device)
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.state.distributed_type == DistributedType.TPU:
            xm.optimizer_step(self.optimizer)
        elif self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]
