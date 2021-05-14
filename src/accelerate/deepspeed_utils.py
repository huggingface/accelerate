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

from dataclasses import dataclass, field
from deepspeed import DeepSpeedEngine

from .optimizer import AcceleratedOptimizer
from .state import is_apex_available


@dataclass
class DeepSpeedPlugin:
    # TODO: need to fix env variables

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of steps to accumulate gradients before updating optimizer states"}
    )
    zero_stage: int = field(default=None, metadata={"help": "Possible options are 0,1,2; Default will be taken from environment variable"})
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"},
    )

    fp16: bool = field(repr=False, default=False, metadata={"help": "You need not define this here"})

    def __post_init__(self):
        self.ds_config = {
            "train_batch_size": None,
            "fp16": {"enabled": False if self.fp16 is None else self.fp16},
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "zero_optimization": {"stage": self.zero_stage},
            "steps_per_print": float("inf"),  # this will stop deepspeed from logging @ stdout
        }


class DeepSpeedEngineWrapper(DeepSpeedEngine):
    """
    Wrapper over deepspeed.DeepSpeedEngine object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # overwriting micro_steps for user's gradient_accumulation
        self.micro_steps = -1

    def step(self, lr_kwargs=None):
        """ DeepSpeedEngine.step() without `micro_steps` update & no profiling """
        if self.is_gradient_accumulation_boundary():  # it shouldn't matter whether we keep this line or not
            if self.progressive_layer_drop:
                self.progressive_layer_drop.update_state(self.global_steps)

            self._take_model_step(lr_kwargs)

    def backward(self, loss):
        """ DeepSpeedEngine.backward() with no loss scaling & no profiling """
        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
            self.optimizer.backward(loss)
        elif self.amp_enabled():
            assert (
                is_apex_available
            ), "You have enabled apex in deepspeed_plugin, but apex is unavailable in your machine"
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
        elif self.fp16_enabled():
            self.optimizer.backward(loss)
        else:
            loss.backward()

        # this will ensure deepspeed gradient_accumulation matches user's accumulation
        self.micro_steps += 1


class DeepSpeedOptimizerWrapper(AcceleratedOptimizer):
    """
    Internal wrapper around a deepspeed optimizer.

    Args:
        optimizer (:obj:`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer, model: DeepSpeedEngineWrapper):
        super().__init__(optimizer, device_placement=False, scaler=None)

        self.model = model

    def zero_grad(self, set_to_none=None):
        """ `model.step()` is doing that automatically. Therefore, it's implementation is not needed """

    def step(self):
        """ This will handle optimizer.step() & optimizer.zero_grad() with gradient_accumulation """
        self.model.step()

    @property
    def is_overflow(self):
        """ This must be called before lr_scheduler.step() when using deepspeed with fp16 """
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        return overflow
