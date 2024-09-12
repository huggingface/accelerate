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

import base64
import json
import os
from copy import deepcopy

from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .dataclasses import DistributedType


def get_active_deepspeed_plugin(state):
    """
    Returns the currently active DeepSpeedPlugin.

    Raises:
        ValueError: If DeepSpeed was not enabled and this function is called.
    """
    if state.distributed_type != DistributedType.DEEPSPEED:
        raise ValueError(
            "Couldn't retrieve the active `DeepSpeedPlugin` as none were enabled. "
            "Please make sure that either `Accelerator` is configured for `deepspeed` "
            "or make sure that the desired `DeepSpeedPlugin` has been enabled (`AcceleratorState().select_deepspeed_plugin(name)`) "
            "before calling this function."
        )
    if not isinstance(state.deepspeed_plugins, dict):
        return state.deepspeed_plugins
    return next(plugin for plugin in state.deepspeed_plugins.values() if plugin.selected)


class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            config = deepcopy(config_file_or_dict)
        elif os.path.exists(config_file_or_dict):
            with open(config_file_or_dict, encoding="utf-8") as f:
                config = json.load(f)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config_file_or_dict).decode("utf-8")
                config = json.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError, ValueError):
                raise ValueError(
                    f"Expected a string path to an existing deepspeed config, or a dictionary, or a base64 encoded string. Received: {config_file_or_dict}"
                )

        self.config = config

        self.set_stage_and_offload()

    def set_stage_and_offload(self):
        # zero stage - this is done as early as possible, before model is created, to allow
        # ``is_deepspeed_zero3_enabled`` query and getting to the early deepspeed config object
        # during ``zero.Init()`` which needs to know the dtype, and some other hparams.
        self._stage = self.get_value("zero_optimization.stage", -1)

        # offload
        self._offload = False
        if self.is_zero2() or self.is_zero3():
            offload_devices_valid = set(["cpu", "nvme"])
            offload_devices = set(
                [
                    self.get_value("zero_optimization.offload_optimizer.device"),
                    self.get_value("zero_optimization.offload_param.device"),
                ]
            )
            if len(offload_devices & offload_devices_valid) > 0:
                self._offload = True

    def find_config_node(self, ds_key_long):
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        """
        Returns the set value or `default` if no value is set
        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def del_config_sub_tree(self, ds_key_long, must_exist=False):
        """
        Deletes a sub-section of the config file if it's found.

        Unless `must_exist` is `True` the section doesn't have to exist.
        """
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        for node in nodes:
            parent_config = config
            config = config.get(node)
            if config is None:
                if must_exist:
                    raise ValueError(f"Can't find {ds_key_long} entry in the config: {self.config}")
                else:
                    return

        # if found remove it
        if parent_config is not None:
            parent_config.pop(node)

    def is_true(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `True` (and it's not set to `False`` or isn't set).

        """
        value = self.get_value(ds_key_long)
        return False if value is None else bool(value)

    def is_false(self, ds_key_long):
        """
        Returns `True`/``False` only if the value is set, always `False` otherwise. So use this method to ask the very
        specific question of whether the value is set to `False` (and it's not set to `True`` or isn't set).
        """
        value = self.get_value(ds_key_long)
        return False if value is None else not bool(value)

    def is_zero2(self):
        return self._stage == 2

    def is_zero3(self):
        return self._stage == 3

    def is_offload(self):
        return self._offload


class DeepSpeedEngineWrapper:
    """
    Internal wrapper for deepspeed.runtime.engine.DeepSpeedEngine. This is used to follow conventional training loop.

    Args:
        engine (deepspeed.runtime.engine.DeepSpeedEngine): deepspeed engine to wrap
    """

    def __init__(self, engine):
        self.engine = engine

    def backward(self, loss, **kwargs):
        # runs backpropagation and handles mixed precision
        self.engine.backward(loss, **kwargs)

        # Deepspeed's `engine.step` performs the following operations:
        # - gradient accumulation check
        # - gradient clipping
        # - optimizer step
        # - zero grad
        # - checking overflow
        # - lr_scheduler step (only if engine.lr_scheduler is not None)
        self.engine.step()
        # and this plugin overrides the above calls with no-ops when Accelerate runs under
        # Deepspeed, but allows normal functionality for non-Deepspeed cases thus enabling a simple
        # training loop that works transparently under many training regimes.


class DeepSpeedOptimizerWrapper(AcceleratedOptimizer):
    """
    Internal wrapper around a deepspeed optimizer.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)
        self.__has_overflow__ = hasattr(self.optimizer, "overflow")

    def zero_grad(self, set_to_none=None):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed

    def step(self):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        if self.__has_overflow__:
            return self.optimizer.overflow
        return False


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
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed


class DummyOptim:
    """
    Dummy optimizer presents model parameters or param groups, this is primarily used to follow conventional training
    loop when optimizer config is specified in the deepspeed config file.

    Args:
        lr (float):
            Learning rate.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float):
            Weight decay.
        **kwargs (additional keyword arguments, *optional*):
            Other arguments.
    """

    def __init__(self, params, lr=0.001, weight_decay=0, **kwargs):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs


class DummyScheduler:
    """
    Dummy scheduler presents model parameters or param groups, this is primarily used to follow conventional training
    loop when scheduler config is specified in the deepspeed config file.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        total_num_steps (int, *optional*):
            Total number of steps.
        warmup_num_steps (int, *optional*):
            Number of steps for warmup.
        lr_scheduler_callable (callable, *optional*):
            A callable function that creates an LR Scheduler. It accepts only one argument `optimizer`.
        **kwargs (additional keyword arguments, *optional*):
            Other arguments.
    """

    def __init__(self, optimizer, total_num_steps=None, warmup_num_steps=0, lr_scheduler_callable=None, **kwargs):
        self.optimizer = optimizer
        self.total_num_steps = total_num_steps
        self.warmup_num_steps = warmup_num_steps
        self.lr_scheduler_callable = lr_scheduler_callable
        self.kwargs = kwargs
