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

"""
General namespace and dataclass related classes
"""

import copy
import enum
import functools
import io
import json
import os
import typing
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Iterable, Optional
import warnings

import torch


class KwargsHandler:
    """
    Internal mixin that implements a `to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how your model is wrapped in a
    `torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
    [wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
    information on each argument.

    <Tip warning={true}>

    `gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.

    </Tip>"""

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `GradScaler` is only available in PyTorch 1.5.0 and later versions.

    </Tip>"""

    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True


@dataclass
class InitProcessGroupKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the distributed processes. Please refer
    to the documentation of this
    [method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
    information on each argument.
    """

    init_method: Optional[str] = None
    timeout: timedelta = timedelta(seconds=1800)


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    TPU = "TPU"


class SageMakerDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **DATA_PARALLEL** -- using sagemaker distributed data parallelism.
        - **MODEL_PARALLEL** -- using sagemaker distributed model parallelism.
    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    DATA_PARALLEL = "DATA_PARALLEL"
    MODEL_PARALLEL = "MODEL_PARALLEL"


class ComputeEnvironment(str, enum.Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """

    # Subclassing str as well as Enum allows the `ComputeEnvironment` to be JSON-serializable out of the box.
    LOCAL_MACHINE = "LOCAL_MACHINE"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"


class EnumWithContains(enum.EnumMeta):
    "A metaclass that adds the ability to check if `self` contains an item with the `in` operator"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    "An enum class that can get the value of an item with `str(Enum.key)`"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        "Method to list all the possible items in `cls`"
        return list(map(lambda item: str(item), cls))


class LoggerType(BaseEnum):
    ALL = "all"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"


class PrecisionType(BaseEnum):
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class RNGType(BaseEnum):
    TORCH = "torch"
    CUDA = "cuda"
    XLA = "xla"
    GENERATOR = "generator"


# data classes


@dataclass
class TensorInformation:
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class DeepSpeedPlugin:

    config_file: str = field(default=None, metadata={"help": "Path to the DeepSpeed config file."})
    gradient_accumulation_steps: int = field(
        default=None, metadata={"help": "Number of steps to accumulate gradients before updating optimizer states"}
    )
    gradient_clipping: float = field(default=None, metadata={"help": "Enable gradient clipping with value"})
    zero_stage: int = field(
        default=None,
        metadata={"help": "Possible options are 0,1,2,3; Default will be taken from environment variable"},
    )
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"},
    )
    offload_optimizer_device: bool = field(
        default=None,
        metadata={
            "help": "Possible options are none|cpu|nvme.\
        Only applicable with ZeRO Stages 2 and 3."
        },
    )
    offload_param_device: bool = field(
        default=None,
        metadata={
            "help": "Possible options are none|cpu|nvme.\
        Only applicable with ZeRO Stage 3."
        },
    )
    zero3_init_flag: bool = field(
        default=None,
        metadata={
            "help": "Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models.\
        Only applicable with ZeRO Stage-3."
        },
    )
    zero3_save_16bit_model: bool = field(
        default=None,
        metadata={
            "help": "Flag to indicate whether to save 16-bit model.\
        Only applicable with ZeRO Stage-3."
        },
    )

    def __post_init__(self):
        if self.config_file is None:
            self.config_file = os.environ.get("DEEPSPEED_CONFIG_FILE", "none")
        if self.config_file != "none":
            with io.open(self.config_file, "r", encoding="utf-8") as f:
                self.deepspeed_config = json.load(f)
            if "gradient_accumulation_steps" not in self.deepspeed_config:
                self.deepspeed_config["gradient_accumulation_steps"] = 1
            if "zero_optimization" not in self.deepspeed_config:
                raise ValueError("Please specify the ZeRO optimization config in the DeepSpeed config file.")
        else:
            if self.gradient_accumulation_steps is None:
                self.gradient_accumulation_steps = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))

            if self.gradient_clipping is None:
                gradient_clipping = os.environ.get("GRADIENT_CLIPPING", "none")
                if gradient_clipping != "none":
                    self.gradient_clipping = float(gradient_clipping)

            if self.zero_stage is None:
                self.zero_stage = int(os.environ.get("DEEPSPEED_ZERO_STAGE", 2))

            if self.offload_optimizer_device is None:
                self.offload_optimizer_device = os.environ.get("DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", "none")

            if self.offload_param_device is None:
                self.offload_param_device = os.environ.get("DEEPSPEED_OFFLOAD_PARAM_DEVICE", "none")

            if self.zero3_save_16bit_model is None:
                self.zero3_save_16bit_model = os.environ.get("DEEPSPEED_ZERO3_SAVE_16BIT_MODEL", "false") == "true"

            self.deepspeed_config = {
                "train_batch_size": None,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "zero_optimization": {
                    "stage": self.zero_stage,
                    "offload_optimizer": {
                        "device": self.offload_optimizer_device,
                    },
                    "offload_param": {
                        "device": self.offload_param_device,
                    },
                    "stage3_gather_16bit_weights_on_model_save": self.zero3_save_16bit_model,
                },
            }
            if self.gradient_clipping:
                self.deepspeed_config["gradient_clipping"] = self.gradient_clipping
        self.deepspeed_config["steps_per_print"] = float("inf")  # this will stop deepspeed from logging @ stdout
        if self.zero3_init_flag is None:
            self.zero3_init_flag = os.environ.get("DEEPSPEED_ZERO3_INIT", "false") == "true"
        if self.zero3_init_flag and self.deepspeed_config["zero_optimization"]["stage"] != 3:
            warnings.warn("DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.")
            self.zero3_init_flag = False

    def find_config_node(self, ds_key_long):
        config = self.deepspeed_config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def fill_match(self, ds_key_long, mismatches, must_match=True, **kwargs):
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            if ds_key_long in kwargs:
                config[ds_key] = kwargs[ds_key_long]
                return
            else:
                raise ValueError(
                    f"{ds_key_long} not found in arguments.\
                    Please specify {ds_key_long} in the DeepSpeed config file or {ds_key_long} as an argument."
                )

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_key_long in kwargs:
            if ds_val != kwargs[ds_key_long]:
                mismatches.append(f"- ds {ds_key_long}={ds_val} vs arg {ds_key}={kwargs[ds_key_long]}")

    def deepspeed_config_process(self, prefix="", config=None, **kwargs):
        """Process the DeepSpeed config with the values from the kwargs."""

        mismatches = []
        if config is None:
            config = self.deepspeed_config
        for key, value in config.items():
            if isinstance(value, dict):
                self.deepspeed_config_process(prefix=prefix + key + ".", config=value, **kwargs)
            elif isinstance(value, str):
                self.fill_match(prefix + key, mismatches, value, **kwargs)
        if len(mismatches) > 0:
            mismatches = "\n".join(mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch kwargs "
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )


@dataclass
class FullyShardedDataParallelPlugin:
    """
    This plugin is used to enable fully sharded data parallelism.
    """

    sharding_strategy: "typing.Any" = field(
        default=None,
        metadata={"help": "Possible options are [1] FULL_SHARD, [2] SHARD_GRAD_OP"},
    )
    backward_prefetch: "typing.Any" = field(
        default=None,
        metadata={"help": "Possible options are [1] BACKWARD_PRE, [2] BACKWARD_POST"},
    )
    auto_wrap_policy: "typing.Any" = field(
        default=None,
        metadata={"help": "A callable specifying a policy to recursively wrap layers with FSDP"},
    )
    cpu_offload: Optional[Callable] = field(
        default=None,
        metadata={"help": "Decides Whether to offload parameters and gradients to CPU."},
    )
    min_num_params: int = field(
        default=None, metadata={"help": "FSDP's minimum number of parameters for Default Auto Wrapping."}
    )
    ignored_modules: Optional[Iterable[torch.nn.Module]] = field(
        default=None,
        metadata={"help": "A list of modules to ignore for FSDP."},
    )

    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, ShardingStrategy
        from torch.distributed.fsdp.wrap import default_auto_wrap_policy

        if self.sharding_strategy is None:
            self.sharding_strategy = ShardingStrategy(int(os.environ.get("FSDP_SHARDING_STRATEGY", 1)))

        if self.cpu_offload is None:
            if os.environ.get("FSDP_OFFLOAD_PARAMS", "false") == "true":
                self.cpu_offload = CPUOffload(offload_params=True)
            else:
                self.cpu_offload = CPUOffload(offload_params=False)

        if self.min_num_params is None:
            self.min_num_params = int(os.environ.get("FSDP_MIN_NUM_PARAMS", 0))

        if self.auto_wrap_policy is None:
            if self.min_num_params > 0:
                self.auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=self.min_num_params)
