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

import torch

from .logging import get_logger
from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_lomo_available, is_torch_xla_available


logger = get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr


def _get_fused_foreach_supported_devices():
    """
    Returns the device types supporting fused / foreach optimizer kernels, using torch's own tables when available.
    """
    try:
        from torch.utils._foreach_utils import (
            _get_foreach_kernels_supported_devices,
            _get_fused_kernels_supported_devices,
        )

        return _get_fused_kernels_supported_devices(), _get_foreach_kernels_supported_devices()
    except ImportError:
        return ["cuda", "xpu", "hpu", "cpu"], ["cuda", "xpu", "hpu", "mps", "cpu"]


def _apply_fused_optimizer_defaults(optimizer):
    """
    Enable `fused` (or `foreach` as a fallback) multi-tensor optimizer kernels on vanilla `torch.optim` optimizers
    when the parameters can take advantage of them.

    Fused/foreach implementations apply the update across all parameters at once and can speed up optimizer steps
    substantially (~10-30%) compared to the default per-parameter loop, at no accuracy cost. This is most useful
    for the very common pattern of creating the optimizer *before* the model is moved to the accelerator device
    (e.g. `optimizer = torch.optim.AdamW(model.parameters())` followed by `accelerator.prepare(model, optimizer)`):
    torch's own auto-defaults evaluate the parameter devices at construction time, see the CPU-resident parameters
    and conservatively fall back to the slow single-tensor path. By the time `AcceleratedOptimizer` wraps the
    optimizer, parameters have been placed on the right device, so we can safely enable the fast path.

    We only touch optimizers known to support the flags (Adam, AdamW, SGD, Adamax, NAdam, RAdam, RMSprop) and only
    when the user didn't already configure `fused`/`foreach` themselves. Mirrors the safety checks of
    `torch.utils._foreach_utils._default_to_fused_or_foreach`.
    """
    if torch.jit.is_scripting():
        return
    if not isinstance(
        optimizer,
        (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.Adamax, torch.optim.NAdam, torch.optim.RAdam, torch.optim.RMSprop),
    ):
        # Third-party optimizers (bitsandbytes, schedulefree, apex, ...) are left untouched
        return

    params = [p for group in optimizer.param_groups for p in group["params"]]
    if not params:
        return
    # Fused/foreach kernels require every parameter to live on the same supported device
    device_types = {p.device.type for p in params}
    if len(device_types) != 1:
        return
    device_type = next(iter(device_types))
    if any(p.is_sparse for p in params):
        return
    if optimizer.defaults.get("differentiable", False):
        return
    if any(group.get("fused") is not None or group.get("foreach") is not None for group in optimizer.param_groups):
        # User (or torch itself) explicitly configured it; torch copies `defaults` keys into every
        # param group with value None, so we must check the *values* and not key presence.
        return

    fused_devices, foreach_devices = _get_fused_foreach_supported_devices()
    # Mirrors `torch.utils._foreach_utils._default_to_fused_or_foreach`: fused requires all params to be non-sparse
    # tensors (floating point for most optimizers), foreach only requires them to be plain tensors. Note the device
    # capability lists are independent (e.g. mps/cpu support fused but not foreach kernels).
    differentiable = optimizer.defaults.get("differentiable", False)
    fused_capable = device_type in fused_devices and not differentiable and all(
        type(p) in (torch.Tensor, torch.nn.Parameter) and torch.is_floating_point(p) for p in params
    )
    foreach_capable = device_type in foreach_devices and all(type(p) in (torch.Tensor, torch.nn.Parameter) for p in params)

    key = "fused" if fused_capable else "foreach" if foreach_capable else None
    if key is None:
        return
    try:
        optimizer.defaults = {**optimizer.defaults, key: True}
        for group in optimizer.param_groups:
            # torch copies the `defaults` (including `key: None`) into every param group at construction time, so
            # an existing `None` entry means "not configured" and must be overwritten, not skipped.
            group[key] = True
    except Exception:
        # Never break a working training run over an optimization
        logger.warning(f"Could not enable `{key}` optimizer kernels, falling back to the default implementation.")


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
        scaler (`torch.amp.GradScaler` or `torch.cuda.amp.GradScaler`, *optional*):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, device_placement=True, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.accelerator_state = AcceleratorState()
        self.gradient_state = GradientState()
        self.device_placement = device_placement
        self._is_overflow = False

        if self.scaler is not None:
            self._accelerate_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)

        # Handle device placement
        if device_placement:
            state_dict = self.optimizer.state_dict()
            if self.accelerator_state.distributed_type == DistributedType.XLA:
                xm.send_cpu_data_to_device(state_dict, self.accelerator_state.device)
            else:
                state_dict = move_to_device(state_dict, self.accelerator_state.device)
            self.optimizer.load_state_dict(state_dict)

        self._callback_handler = None
        self._accelerator_ref = None
        if device_placement:
            _apply_fused_optimizer_defaults(self.optimizer)

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
        if self.accelerator_state.distributed_type == DistributedType.XLA and self.device_placement:
            xm.send_cpu_data_to_device(state_dict, self.accelerator_state.device)
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if self.gradient_state.sync_gradients:
            accept_arg = "set_to_none" in inspect.signature(self.optimizer.zero_grad).parameters
            if accept_arg:
                if set_to_none is None:
                    set_to_none = True
                self.optimizer.zero_grad(set_to_none=set_to_none)
            else:
                if set_to_none is not None:
                    raise ValueError("`set_to_none` for Optimizer.zero_grad` is not supported by this optimizer.")
                self.optimizer.zero_grad()

    def train(self):
        """
        Sets the optimizer to "train" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        elif (
            hasattr(self.optimizer, "optimizer")
            and hasattr(self.optimizer.optimizer, "train")
            and callable(self.optimizer.optimizer.train)
        ):
            # the deepspeed optimizer further wraps the optimizer
            self.optimizer.optimizer.train()

    def eval(self):
        """
        Sets the optimizer to "eval" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        elif (
            hasattr(self.optimizer, "optimizer")
            and hasattr(self.optimizer.optimizer, "eval")
            and callable(self.optimizer.optimizer.eval)
        ):
            # the deepspeed optimizer further wraps the optimizer
            self.optimizer.optimizer.eval()

    def step(self, closure=None):
        """
        Performs the optimizer step when gradients should be synchronized, firing `on_optimizer_step` callbacks
        exactly once per real step (i.e. skipped during gradient accumulation sub-steps).
        """
        if self.gradient_state.sync_gradients and self._callback_handler is not None:
            self._step(closure)
            self._callback_handler.call_event("on_optimizer_step", self._accelerator_ref)
        else:
            self._step(closure)

    def _step(self, closure=None):
        if is_lomo_available():
            from lomo_optim import AdaLomo, Lomo

        if (
            not self.gradient_state.is_xla_gradients_synced
            and self.accelerator_state.distributed_type == DistributedType.XLA
        ):
            gradients = xm._fetch_gradients(self.optimizer)
            xm.all_reduce("sum", gradients, scale=1.0 / xr.world_size())
            self.gradient_state.is_xla_gradients_synced = True

        if is_lomo_available():
            #  `step` should be a no-op for LOMO optimizers.
            if isinstance(self.optimizer, (Lomo, AdaLomo)):
                return

        if self.gradient_state.sync_gradients:
            if self.scaler is not None:
                self.optimizer.step = self._optimizer_patched_step_method

                self.scaler.step(self.optimizer, closure)
                self.scaler.update()

                if not self._accelerate_step_called:
                    # If the optimizer step was skipped, gradient overflow was detected.
                    self._is_overflow = True
                else:
                    self._is_overflow = False
                # Reset the step method to the original one
                self.optimizer.step = self._optimizer_original_step_method
                # Reset the indicator
                self._accelerate_step_called = False
            else:
                self.optimizer.step(closure)
        if self.accelerator_state.distributed_type == DistributedType.XLA:
            self.gradient_state.is_xla_gradients_synced = False

    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was skipped."""
        return self._is_overflow

    def __getstate__(self):
        _ignored_keys = [
            "_accelerate_step_called",
            "_optimizer_original_step_method",
            "_optimizer_patched_step_method",
            "_callback_handler",
            "_accelerator_ref",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in _ignored_keys}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._callback_handler = None
        self._accelerator_ref = None
        if self.scaler is not None:
            self._accelerate_step_called = False
            self._optimizer_original_step_method = self.optimizer.step
            self._optimizer_patched_step_method = patch_optimizer_step(self, self.optimizer.step)


def patch_optimizer_step(accelerated_optimizer: AcceleratedOptimizer, method):
    def patched_step(*args, **kwargs):
        accelerated_optimizer._accelerate_step_called = True
        return method(*args, **kwargs)

    return patched_step
