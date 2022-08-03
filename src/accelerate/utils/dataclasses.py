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
import os
import typing
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Iterable, Optional

import torch

from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_STATE_DICT_TYPE, MODEL_NAME, OPTIMIZER_NAME


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
    MPS = "MPS"


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
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
    """

    ALL = "all"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"


class PrecisionType(BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """

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
    """
    This plugin is used to integrate DeepSpeed.
    """

    hf_ds_config: Any = field(
        default=None,
        metadata={
            "help": "path to DeepSpeed config file or dict or an object of class `accelerate.utils.deepspeed.HfDeepSpeedConfig`."
        },
    )
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
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3."},
    )
    offload_param_device: bool = field(
        default=None,
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3."},
    )
    zero3_init_flag: bool = field(
        default=None,
        metadata={
            "help": "Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
            "Only applicable with ZeRO Stage-3."
        },
    )
    zero3_save_16bit_model: bool = field(
        default=None,
        metadata={"help": "Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3."},
    )

    def __post_init__(self):
        from .deepspeed import HfDeepSpeedConfig

        if self.hf_ds_config is None:
            self.hf_ds_config = os.environ.get("DEEPSPEED_CONFIG_FILE", "none")
        if (
            isinstance(self.hf_ds_config, dict)
            or (isinstance(self.hf_ds_config, str) and self.hf_ds_config != "none")
            or isinstance(self.hf_ds_config, HfDeepSpeedConfig)
        ):
            if not isinstance(self.hf_ds_config, HfDeepSpeedConfig):
                self.hf_ds_config = HfDeepSpeedConfig(self.hf_ds_config)
            if "gradient_accumulation_steps" not in self.hf_ds_config.config:
                self.hf_ds_config.config["gradient_accumulation_steps"] = 1
            elif self.hf_ds_config.config["gradient_accumulation_steps"] == "auto":
                raise ValueError("gradient_accumulation_steps cannot be set to 'auto' in the DeepSpeed config.")
            if "zero_optimization" not in self.hf_ds_config.config:
                raise ValueError("Please specify the ZeRO optimization config in the DeepSpeed config.")
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

            config = {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
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
                config["gradient_clipping"] = self.gradient_clipping
            self.hf_ds_config = HfDeepSpeedConfig(config)
        self.deepspeed_config = self.hf_ds_config.config
        self.deepspeed_config["steps_per_print"] = float("inf")  # this will stop deepspeed from logging @ stdout
        if self.zero3_init_flag is None:
            self.zero3_init_flag = os.environ.get("DEEPSPEED_ZERO3_INIT", "false") == "true"
        if self.zero3_init_flag and not self.hf_ds_config.is_zero3():
            warnings.warn("DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.")
            self.zero3_init_flag = False

    def fill_match(self, ds_key_long, mismatches, must_match=True, **kwargs):
        config, ds_key = self.hf_ds_config.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            if ds_key_long in kwargs:
                config[ds_key] = kwargs[ds_key_long]
                return
            else:
                raise ValueError(
                    f"`{ds_key_long}` not found in kwargs. "
                    f"Please specify `{ds_key_long}` without `auto`(set to correct value) in the DeepSpeed config file or "
                    "pass it in kwargs."
                )

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_key_long in kwargs:
            if ds_val != kwargs[ds_key_long]:
                mismatches.append(f"- ds {ds_key_long}={ds_val} vs arg {ds_key_long}={kwargs[ds_key_long]}")

    def deepspeed_config_process(self, prefix="", mismatches=None, config=None, must_match=True, **kwargs):
        """Process the DeepSpeed config with the values from the kwargs."""
        mismatches = [] if mismatches is None else mismatches
        if config is None:
            config = self.deepspeed_config
        for key, value in config.items():
            if isinstance(value, dict):
                self.deepspeed_config_process(
                    prefix=prefix + key + ".", mismatches=mismatches, config=value, must_match=must_match, **kwargs
                )
            else:
                self.fill_match(prefix + key, mismatches, must_match=must_match, **kwargs)
        if len(mismatches) > 0 and prefix == "":
            mismatches_msg = "\n".join(mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch kwargs "
                f" values:\n{mismatches_msg}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )

    def set_mixed_precision(self, mixed_precision):
        ds_config = self.deepspeed_config
        if mixed_precision == "fp16" and "fp16" not in ds_config and "bf16" not in ds_config:
            ds_config.update({"fp16": {"enabled": True, "auto_cast": True}})
        elif mixed_precision == "bf16" and "fp16" not in ds_config and "bf16" not in ds_config:
            ds_config.update({"bf16": {"enabled": True}})

    def set_deepspeed_weakref(self):
        from .imports import is_transformers_available

        if self.zero3_init_flag:
            if not is_transformers_available():
                raise Exception(
                    "When `zero3_init_flag` is set, it requires Transformers to be installed. "
                    "Please run `pip install transformers`."
                )
            ds_config = copy.deepcopy(self.deepspeed_config)
            if "gradient_accumulation_steps" not in ds_config or ds_config["gradient_accumulation_steps"] == "auto":
                ds_config["gradient_accumulation_steps"] = 1
            if (
                "train_micro_batch_size_per_gpu" not in ds_config
                or ds_config["train_micro_batch_size_per_gpu"] == "auto"
            ):
                ds_config["train_micro_batch_size_per_gpu"] = 1
            if ds_config["train_batch_size"] == "auto":
                del ds_config["train_batch_size"]

            from transformers.deepspeed import HfDeepSpeedConfig

            self.dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive # noqa


@dataclass
class FullyShardedDataParallelPlugin:
    """
    This plugin is used to enable fully sharded data parallelism.
    """

    sharding_strategy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Sharding Strategy of type `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`"
        },
    )
    backward_prefetch: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Backward Prefetch of type `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`"
        },
    )
    mixed_precision_policy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "A config to enable mixed precision training with FullyShardedDataParallel. "
            "The 3 flags that are set are `param_dtype`, `reduce_dtype`, `buffer_dtype`. "
            "Each flag expects `torch.dtype` as the value. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.MixedPrecision`."
        },
    )
    auto_wrap_policy: Optional[Callable] = field(
        default=None,
        metadata={"help": "A callable specifying a policy to recursively wrap layers with FSDP"},
    )
    cpu_offload: "typing.Any" = field(
        default=None,
        metadata={
            "help": "Decides Whether to offload parameters and gradients to CPU. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload`."
        },
    )
    ignored_modules: Optional[Iterable[torch.nn.Module]] = field(
        default=None,
        metadata={"help": "A list of modules to ignore for FSDP."},
    )

    state_dict_type: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Type of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictType`"
        },
    )

    state_dict_config: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Config of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictConfig`"
        },
    )

    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            BackwardPrefetch,
            CPUOffload,
            ShardingStrategy,
            StateDictType,
            _state_dict_type_to_config,
        )

        if self.sharding_strategy is None:
            self.sharding_strategy = ShardingStrategy(int(os.environ.get("FSDP_SHARDING_STRATEGY", 1)))

        if self.cpu_offload is None:
            if os.environ.get("FSDP_OFFLOAD_PARAMS", "false") == "true":
                self.cpu_offload = CPUOffload(offload_params=True)
            else:
                self.cpu_offload = CPUOffload(offload_params=False)

        if self.backward_prefetch is None:
            prefetch_policy = os.environ.get("FSDP_BACKWARD_PREFETCH", "NO_PREFETCH")
            if prefetch_policy != FSDP_BACKWARD_PREFETCH[-1]:
                self.backward_prefetch = BackwardPrefetch(FSDP_BACKWARD_PREFETCH.index(prefetch_policy) + 1)

        if self.state_dict_type is None:
            state_dict_type_policy = os.environ.get("FSDP_STATE_DICT_TYPE", "FULL_STATE_DICT")
            self.state_dict_type = StateDictType(FSDP_STATE_DICT_TYPE.index(state_dict_type_policy) + 1)

            if self.state_dict_type == StateDictType.FULL_STATE_DICT:
                self.state_dict_config = _state_dict_type_to_config[self.state_dict_type](
                    offload_to_cpu=True, rank0_only=True
                )
            else:
                self.state_dict_config = _state_dict_type_to_config[self.state_dict_type]()

    @staticmethod
    def get_module_class_from_name(module, name):
        """
        Gets a class from a module by its name.

        Args:
            module (`torch.nn.Module`): The module to get the class from.
            name (`str`): The name of the class.
        """
        modules_children = list(module.children())
        if module.__class__.__name__ == name:
            return module.__class__
        elif len(modules_children) == 0:
            return
        else:
            for child_module in modules_children:
                module_class = FullyShardedDataParallelPlugin.get_module_class_from_name(child_module, name)
                if module_class is not None:
                    return module_class

    def set_auto_wrap_policy(self, model):
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

        if self.auto_wrap_policy is None:
            auto_wrap_policy = os.environ.get("FSDP_AUTO_WRAP_POLICY", "NO_WRAP")
            if auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[0]:
                transformer_cls_to_wrap = os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", "")
                transformer_cls_to_wrap = FullyShardedDataParallelPlugin.get_module_class_from_name(
                    model, transformer_cls_to_wrap
                )
                if transformer_cls_to_wrap is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                self.auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls={transformer_cls_to_wrap},
                )
            elif auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[1]:
                min_num_params = int(os.environ.get("FSDP_MIN_NUM_PARAMS", 0))
                if min_num_params > 0:
                    self.auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=min_num_params
                    )

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == "fp16":
            dtype = torch.float16
        elif mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown mixed precision value: {mixed_precision}")
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

        if self.mixed_precision_policy is None:
            self.mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def save_model(self, accelerator, model, output_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            with FSDP.state_dict_type(model, self.state_dict_type, self.state_dict_config):
                state_dict = model.state_dict()
            weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                print(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                print(f"Model saved to {output_model_file}")
        else:
            with FSDP.state_dict_type(model, self.state_dict_type, self.state_dict_config):
                state_dict = model.state_dict()
            weights_name = (
                f"{MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            output_model_file = os.path.join(output_dir, weights_name)
            print(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            print(f"Model saved to {output_model_file}")

    def load_model(self, accelerator, model, input_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        accelerator.wait_for_everyone()

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            accelerator.print(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            accelerator.print(f"Model loaded from {input_model_file}")
        else:
            weights_name = (
                f"{MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            print(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            print(f"Model loaded from {input_model_file}")
        with FSDP.state_dict_type(model, self.state_dict_type, self.state_dict_config):
            model.load_state_dict(state_dict)

    def save_optimizer(self, accelerator, optimizer, model, output_dir, optimizer_index=0, optim_input=None):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        optim_state = FSDP.full_optim_state_dict(model, optimizer, optim_input=optim_input)
        if accelerator.process_index == 0:
            optim_state_name = (
                f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
            )
            output_optimizer_file = os.path.join(output_dir, optim_state_name)
            print(f"Saving Optimizer state to {output_optimizer_file}")
            torch.save(optim_state, output_optimizer_file)
            print(f"Optimizer state saved in {output_optimizer_file}")

    def load_optimizer(self, accelerator, optimizer, model, input_dir, optimizer_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        accelerator.wait_for_everyone()
        full_osd = None
        if accelerator.process_index == 0:
            optimizer_name = (
                f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
            )
            input_optimizer_file = os.path.join(input_dir, optimizer_name)
            print(f"Loading Optimizer state from {input_optimizer_file}")
            full_osd = torch.load(input_optimizer_file)
            print(f"Optimizer state loaded from {input_optimizer_file}")
        # called from all ranks, though only rank0 has a valid param for full_osd
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
        optimizer.load_state_dict(sharded_osd)
