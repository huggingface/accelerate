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
import torch.nn as nn
from dataclasses import dataclass, field

try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False


@dataclass
class DeepSpeedPlugin:

    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before updating optimizer states"})
    zero_stage: int = field(default=0, metadata={"help": "Available options are 0,1"})
    is_train_batch_min: str = field(default=True, metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"})

    fp16: bool = field(repr=False, default=False, metadata={"help": "You need not define this here"})

    def __post_init__(self):
        self.ds_config = {
            "train_batch_size": None,
            "fp16": {"enabled": False if self.fp16 is None else self.fp16},
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "zero_optimization": {"stage": self.zero_stage},
        }


class DeepSpeedEngineWrapper(nn.Module):
    """
    Internal wrapper around DeepSpeedEngine instance

    Args:
        model (:obj: `DeepSpeedEngine`):
            DeepSpeedEngine instance created using deepspeed.initalize()
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.optimizer = model.optimizer

        self.progressive_layer_drop = model.progressive_layer_drop
        self.global_steps = model.global_steps

        # overwriting micro_steps for user's gradient_accumulation
        self.model.micro_steps = -1

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def is_gradient_accumulation_boundary(self):
        return self.model.is_gradient_accumulation_boundary()

    def zero_optimization(self):
        return self.model.zero_optimization()

    def amp_enabled(self):
        return self.model.amp_enabled()

    def fp16_enabled(self):
        return self.model.fp16_enabled()

    def _take_model_step(self, *args, **kwargs):
        return self.model._take_model_step(*args, **kwargs)

    def step(self, lr_kwargs=None):
        """ DeepSpeedEngine.step() without `micro_steps` update & no profiling """
        if self.is_gradient_accumulation_boundary(): # it shouldn't matter whether we keep this line or not
            if self.progressive_layer_drop:
                self.progressive_layer_drop.update_state(self.global_steps)

            self._take_model_step(lr_kwargs)

    def backward(self, loss):
        """ DeepSpeedEngine.backward() with no loss scaling & no profiling """
        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary(
            )
            self.optimizer.backward(loss)
        elif self.amp_enabled():
            assert _apex_available, "You have enabled apex in deepspeed_plugin, but apex is unavailable in your machine"
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss,
                                self.optimizer,
                                delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
        elif self.fp16_enabled():
            self.optimizer.backward(loss)
        else:
            loss.backward()

        # this will ensure deepspeed gradient_accumulation matches user's accumulation
        self.model.micro_steps += 1

    def save_checkpoint(self, *args, **kwargs):
        return self.model.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        return self.model.load_checkpoint(*args, **kwargs)


class DeepSpeedOptimizerWrapper(torch.optim.Optimizer):
    """
    Internal wrapper around a deepspeed optimizer.

    Args:
        optimizer (:obj:`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer, model: DeepSpeedEngineWrapper):
        self.optimizer = optimizer
        self.model = model

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
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        """ `model.step()` is doing that automatically. Therefore, it's implementation is not needed """

    def step(self):
        """ This will handle optimizer.step() & optimizer.zero_grad() with gradient_accumulation """
        self.model.step()

    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]

    @property
    def is_overflow(self):
        """ This must be called before lr_scheduler.step() when using deepspeed with fp16 """
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        return overflow
