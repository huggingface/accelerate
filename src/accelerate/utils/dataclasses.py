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

import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from distutils.util import strtobool
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_STATE_DICT_TYPE, MODEL_NAME, OPTIMIZER_NAME
from .versions import is_torch_version


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

    `static_graph` is only available in PyTorch 1.11.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `GradScaler` is only available in PyTorch 1.5.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs

    kwargs = GradScalerKwargs(backoff_filter=0.25)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

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

    ```python
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
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
    MEGATRON_LM = "MEGATRON_LM"


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


class DynamoBackend(str, enum.Enum):
    """
    Represents a dynamo backend (see https://github.com/pytorch/torchdynamo).

    Values:

        - **NO** -- Do not use torch dynamo.
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **NVFUSER** -- nvFuser with TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_NVFUSER** -- nvFuser with AotAutograd. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read
          more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).

    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    EAGER = "EAGER"
    AOT_EAGER = "AOT_EAGER"
    INDUCTOR = "INDUCTOR"
    NVFUSER = "NVFUSER"
    AOT_NVFUSER = "AOT_NVFUSER"
    AOT_CUDAGRAPHS = "AOT_CUDAGRAPHS"
    OFI = "OFI"
    FX2TRT = "FX2TRT"
    ONNXRT = "ONNXRT"
    IPEX = "IPEX"


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
        return list(map(str, cls))


class LoggerType(BaseEnum):
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
    """

    ALL = "all"
    AIM = "aim"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"
    MLFLOW = "mlflow"


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
class ProjectConfiguration:
    """
    Configuration for the Accelerator object based on inner-project needs.
    """

    project_dir: str = field(default=None, metadata={"help": "A path to a directory for storing data."})
    logging_dir: str = field(
        default=None,
        metadata={
            "help": "A path to a directory for storing logs of locally-compatible loggers. If None, defaults to `project_dir`."
        },
    )
    automatic_checkpoint_naming: bool = field(
        default=False,
        metadata={"help": "Whether saved states should be automatically iteratively named."},
    )

    total_limit: int = field(
        default=None,
        metadata={"help": "The maximum number of total saved states to keep."},
    )

    iteration: int = field(
        default=0,
        metadata={"help": "The current save iteration."},
    )

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = self.project_dir


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

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", 1))

        if self.gradient_clipping is None:
            gradient_clipping = os.environ.get("ACCELERATE_GRADIENT_CLIPPING", "none")
            if gradient_clipping != "none":
                self.gradient_clipping = float(gradient_clipping)

        if self.zero_stage is None:
            self.zero_stage = int(os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", 2))

        if self.offload_optimizer_device is None:
            self.offload_optimizer_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", "none")

        if self.offload_param_device is None:
            self.offload_param_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE", "none")

        if self.zero3_save_16bit_model is None:
            self.zero3_save_16bit_model = (
                os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL", "false") == "true"
            )

        if self.hf_ds_config is None:
            self.hf_ds_config = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "none")
        if (
            isinstance(self.hf_ds_config, dict)
            or (isinstance(self.hf_ds_config, str) and self.hf_ds_config != "none")
            or isinstance(self.hf_ds_config, HfDeepSpeedConfig)
        ):
            if not isinstance(self.hf_ds_config, HfDeepSpeedConfig):
                self.hf_ds_config = HfDeepSpeedConfig(self.hf_ds_config)
            if "gradient_accumulation_steps" not in self.hf_ds_config.config:
                self.hf_ds_config.config["gradient_accumulation_steps"] = 1
            if "zero_optimization" not in self.hf_ds_config.config:
                raise ValueError("Please specify the ZeRO optimization config in the DeepSpeed config.")

            self._deepspeed_config_checks()
            kwargs = {
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "gradient_clipping": self.gradient_clipping if self.gradient_clipping else 1.0,
                "zero_optimization.stage": self.zero_stage,
                "zero_optimization.offload_optimizer.device": self.offload_optimizer_device,
                "zero_optimization.offload_param.device": self.offload_param_device,
                "zero_optimization.stage3_gather_16bit_weights_on_model_save": self.zero3_save_16bit_model,
            }
            for key in kwargs.keys():
                self.fill_match(key, **kwargs, must_match=False)
            self.hf_ds_config.set_stage_and_offload()
        else:
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
            self.zero3_init_flag = (
                strtobool(os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_INIT", str(self.hf_ds_config.is_zero3()))) == 1
            )
        if self.zero3_init_flag and not self.hf_ds_config.is_zero3():
            warnings.warn("DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.")
            self.zero3_init_flag = False

    def fill_match(self, ds_key_long, mismatches=None, must_match=True, **kwargs):
        mismatches = [] if mismatches is None else mismatches
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
        kwargs = {
            "fp16.enabled": mixed_precision == "fp16",
            "bf16.enabled": mixed_precision == "bf16",
        }
        if mixed_precision == "fp16":
            if "fp16" not in ds_config:
                ds_config["fp16"] = {"enabled": True, "auto_cast": True}
        elif mixed_precision == "bf16":
            if "bf16" not in ds_config:
                ds_config["bf16"] = {"enabled": True}

        if mixed_precision != "no":
            diff_dtype = "bf16" if mixed_precision == "fp16" else "fp16"
            if str(ds_config.get(diff_dtype, {}).get("enabled", "False")).lower() == "true":
                raise ValueError(
                    f"`--mixed_precision` arg cannot be set to `{mixed_precision}` when `{diff_dtype}` is set in the DeepSpeed config file."
                )
        for dtype in ["fp16", "bf16"]:
            if dtype not in ds_config:
                ds_config[dtype] = {"enabled": False}
        self.fill_match("fp16.enabled", must_match=False, **kwargs)
        self.fill_match("bf16.enabled", must_match=False, **kwargs)

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

    def is_zero3_init_enabled(self):
        return self.zero3_init_flag

    @contextmanager
    def zero3_init_context_manager(self, enable=False):
        old = self.zero3_init_flag
        if old == enable:
            yield
        else:
            self.zero3_init_flag = enable
            self.dschf = None
            self.set_deepspeed_weakref()
            yield
            self.zero3_init_flag = old
            self.dschf = None
            self.set_deepspeed_weakref()

    def _deepspeed_config_checks(self):
        env_variable_names_to_ignore = [
            "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
            "ACCELERATE_GRADIENT_CLIPPING",
            "ACCELERATE_DEEPSPEED_ZERO_STAGE",
            "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE",
            "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE",
            "ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL",
            "ACCELERATE_MIXED_PRECISION",
        ]
        env_variable_names_to_ignore = [
            name.replace("ACCELERATE_", "").replace("DEEPSPEED_", "").lower() for name in env_variable_names_to_ignore
        ]

        deepspeed_fields_from_accelerate_config = os.environ.get("ACCELERATE_CONFIG_DS_FIELDS", "").split(",")

        if any(name in env_variable_names_to_ignore for name in deepspeed_fields_from_accelerate_config):
            raise ValueError(
                f"When using `deepspeed_config_file`, the following accelerate config variables will be ignored: {env_variable_names_to_ignore}.\n"
                "Please specify them appropriately in the DeepSpeed config file.\n"
                "If you are using an accelerate config file, remove others config variables mentioned in the above specified list.\n"
                "The easiest method is to create a new config following the questionnaire via `accelerate config`.\n"
                "It will only ask for the necessary config variables when using `deepspeed_config_file`."
            )


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

    limit_all_gathers: bool = field(
        default=False,
        metadata={
            "help": "If False, then FSDP allows the CPU thread to schedule all-gathers "
            "without any extra synchronization. If True, then FSDP explicitly synchronizes the CPU thread to prevent "
            "too many in-flight all-gathers. This bool only affects the sharded strategies that schedule all-gathers. "
            "Enabling this can help lower the number of CUDA malloc retries."
        },
    )

    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            BackwardPrefetch,
            CPUOffload,
            FullStateDictConfig,
            ShardingStrategy,
            StateDictType,
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

            if self.state_dict_type == StateDictType.FULL_STATE_DICT and self.state_dict_config is None:
                self.state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

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

        if is_torch_version("<=", "1.13.5"):
            with FSDP.state_dict_type(model, self.state_dict_type, self.state_dict_config):
                state_dict = model.state_dict()
        else:
            FSDP.set_state_dict_type(model, self.state_dict_type, self.state_dict_config)
            state_dict = model.state_dict()

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                print(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                print(f"Model saved to {output_model_file}")
        else:
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

        if is_torch_version("<=", "1.13.5"):
            with FSDP.state_dict_type(model, self.state_dict_type, self.state_dict_config):
                model.load_state_dict(state_dict)
        else:
            FSDP.set_state_dict_type(model, self.state_dict_type, self.state_dict_config)
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


@dataclass
class MegatronLMPlugin:
    """
    Plugin for Megatron-LM to enable tensor, pipeline, sequence and data parallelism. Also to enable selective
    activation recomputation and optimized fused kernels.
    """

    tp_degree: int = field(default=None, metadata={"help": "tensor parallelism degree."})
    pp_degree: int = field(default=None, metadata={"help": "pipeline parallelism degree."})
    num_micro_batches: int = field(default=None, metadata={"help": "number of micro-batches."})
    gradient_clipping: float = field(
        default=None, metadata={"help": "gradient clipping value based on global L2 Norm (0 to disable)"}
    )
    sequence_parallelism: bool = field(
        default=None,
        metadata={"help": "enable sequence parallelism"},
    )
    recompute_activation: bool = field(
        default=None,
        metadata={"help": "enable selective activation recomputation"},
    )
    use_distributed_optimizer: bool = field(
        default=None,
        metadata={"help": "enable distributed optimizer"},
    )
    pipeline_model_parallel_split_rank: int = field(
        default=None, metadata={"help": "Rank where encoder and decoder should be split."}
    )
    num_layers_per_virtual_pipeline_stage: int = field(
        default=None, metadata={"help": "Number of layers per virtual pipeline stage."}
    )
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the micro_batch_size"},
    )
    train_iters: int = field(
        default=None,
        metadata={
            "help": "Total number of iterations to train over all training runs. "
            "Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`"
        },
    )
    train_samples: int = field(
        default=None,
        metadata={
            "help": "Total number of samples to train over all training runs. "
            "Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`"
        },
    )
    weight_decay_incr_style: str = field(
        default="constant",
        metadata={"help": 'Weight decay increment function. choices=["constant", "linear", "cosine"]. '},
    )
    start_weight_decay: float = field(
        default=None,
        metadata={"help": "Initial weight decay coefficient for L2 regularization."},
    )
    end_weight_decay: float = field(
        default=None,
        metadata={"help": "End of run weight decay coefficient for L2 regularization."},
    )
    lr_decay_style: str = field(
        default="linear",
        metadata={"help": "Learning rate decay function. choices=['constant', 'linear', 'cosine']."},
    )
    lr_decay_iters: int = field(
        default=None,
        metadata={"help": "Number of iterations for learning rate decay. If None defaults to `train_iters`."},
    )
    lr_decay_samples: int = field(
        default=None,
        metadata={"help": "Number of samples for learning rate decay. If None defaults to `train_samples`."},
    )
    lr_warmup_iters: int = field(
        default=None,
        metadata={"help": "number of iterations to linearly warmup learning rate over."},
    )
    lr_warmup_samples: int = field(
        default=None,
        metadata={"help": "number of samples to linearly warmup learning rate over."},
    )
    lr_warmup_fraction: float = field(
        default=None,
        metadata={"help": "fraction of lr-warmup-(iters/samples) to linearly warmup learning rate over."},
    )
    min_lr: float = field(
        default=0,
        metadata={"help": "Minumum value for learning rate. The scheduler clip values below this threshold."},
    )
    consumed_samples: List[int] = field(
        default=None,
        metadata={
            "help": "Number of samples consumed in the same order as the dataloaders to `accelerator.prepare` call."
        },
    )
    no_wd_decay_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to disable weight decay."})
    scale_lr_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to scale learning rate."})
    lr_mult: float = field(default=1.0, metadata={"help": "Learning rate multiplier."})
    megatron_dataset_flag: bool = field(
        default=False,
        metadata={"help": "Whether the format of dataset follows Megatron-LM Indexed/Cached/MemoryMapped format."},
    )
    seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process."},
    )
    encoder_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process for the encoder."},
    )
    decoder_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process for the decoder."},
    )
    tensorboard_dir: str = field(
        default=None,
        metadata={"help": "Path to save tensorboard logs."},
    )
    set_all_logging_options: bool = field(
        default=False,
        metadata={"help": "Whether to set all logging options."},
    )
    eval_iters: int = field(
        default=100, metadata={"help": "Number of iterations to run for evaluation validation/test for."}
    )
    eval_interval: int = field(
        default=1000, metadata={"help": "Interval between running evaluation on validation set."}
    )
    return_logits: bool = field(
        default=False,
        metadata={"help": "Whether to return logits from the model."},
    )

    # custom train step args
    custom_train_step_class: Optional[Any] = field(
        default=None,
        metadata={"help": "Custom train step class."},
    )
    custom_train_step_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Custom train step kwargs."},
    )

    # custom model args
    custom_model_provider_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom model provider function."},
    )
    custom_prepare_model_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom prepare model function."},
    )

    # remaining args such as enabling Alibi/ROPE positional embeddings,
    # wandb logging, Multi-Query Attention, etc.
    other_megatron_args: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Other Megatron-LM arguments. Please refer Megatron-LM"},
    )

    def __post_init__(self):
        prefix = "MEGATRON_LM_"
        if self.tp_degree is None:
            self.tp_degree = int(os.environ.get(prefix + "TP_DEGREE", 1))
        if self.pp_degree is None:
            self.pp_degree = int(os.environ.get(prefix + "PP_DEGREE", 1))
        if self.num_micro_batches is None:
            self.num_micro_batches = int(os.environ.get(prefix + "NUM_MICRO_BATCHES", 1))
        if self.gradient_clipping is None:
            self.gradient_clipping = float(os.environ.get(prefix + "GRADIENT_CLIPPING", 1.0))
        if self.recompute_activation is None:
            self.recompute_activation = strtobool(os.environ.get(prefix + "RECOMPUTE_ACTIVATION", "False")) == 1
        if self.use_distributed_optimizer is None:
            self.use_distributed_optimizer = (
                strtobool(os.environ.get(prefix + "USE_DISTRIBUTED_OPTIMIZER", "False")) == 1
            )
        if self.sequence_parallelism is None:
            self.sequence_parallelism = strtobool(os.environ.get(prefix + "SEQUENCE_PARALLELISM", "False")) == 1

        if self.pp_degree > 1 or self.use_distributed_optimizer:
            self.DDP_impl = "local"
        else:
            self.DDP_impl = "torch"

        if self.consumed_samples is not None:
            if len(self.consumed_samples) == 1:
                self.consumed_samples.extend([0, 0])
            elif len(self.consumed_samples) == 2:
                self.consumed_samples.append(0)

        self.megatron_lm_default_args = {
            "tensor_model_parallel_size": self.tp_degree,
            "pipeline_model_parallel_size": self.pp_degree,
            "pipeline_model_parallel_split_rank": self.pipeline_model_parallel_split_rank,
            "num_layers_per_virtual_pipeline_stage": self.num_layers_per_virtual_pipeline_stage,
            "DDP_impl": self.DDP_impl,
            "use_distributed_optimizer": self.use_distributed_optimizer,
            "sequence_parallel": self.sequence_parallelism,
            "clip_grad": self.gradient_clipping,
            "num_micro_batches": self.num_micro_batches,
            "consumed_samples": self.consumed_samples,
            "no_wd_decay_cond": self.no_wd_decay_cond,
            "scale_lr_cond": self.scale_lr_cond,
            "lr_mult": self.lr_mult,
            "megatron_dataset_flag": self.megatron_dataset_flag,
            "eval_iters": self.eval_iters,
            "eval_interval": self.eval_interval,
        }
        if self.recompute_activation:
            self.megatron_lm_default_args["recompute_granularity"] = "selective"
        if self.tensorboard_dir is not None:
            self.megatron_lm_default_args["tensorboard_dir"] = self.tensorboard_dir
            if self.set_all_logging_options:
                self.set_tensorboard_logging_options()
        if self.other_megatron_args is not None:
            self.megatron_lm_default_args.update(self.other_megatron_args)

    def set_network_size_args(self, model, batch_data=None):
        # Check if the model is either BERT, GPT or T5 else raise error
        # set 'num_layers', 'hidden_size', 'num_attention_heads', 'max_position_embeddings'
        if "megatron-bert" in model.config.model_type.lower():
            model_type_name = "bert"
            num_layers = model.config.num_hidden_layers
            hidden_size = model.config.hidden_size
            num_attention_heads = model.config.num_attention_heads
            max_position_embeddings = model.config.max_position_embeddings
            num_labels = model.config.num_labels
            orig_vocab_size = model.config.vocab_size
            if "maskedlm" in model.__class__.__name__.lower():
                pretraining_flag = True
            if self.seq_length is not None:
                if self.encoder_seq_length is not None:
                    warnings.warn("Both `seq_length` and `encoder_seq_length` are set. Using `encoder_seq_length`.")
                self.seq_length = self.encoder_seq_length
            elif self.encoder_seq_length is not None:
                self.seq_length = self.encoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data["input_ids"].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args["seq_length"] = self.seq_length
        elif "gpt2" in model.config.model_type.lower():
            model_type_name = "gpt"
            num_layers = model.config.n_layer
            hidden_size = model.config.n_embd
            num_attention_heads = model.config.n_head
            max_position_embeddings = model.config.n_positions
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.seq_length is not None:
                if self.decoder_seq_length is not None:
                    warnings.warn("Both `seq_length` and `decoder_seq_length` are set. Using `decoder_seq_length`.")
                self.seq_length = self.decoder_seq_length
            elif self.decoder_seq_length is not None:
                self.seq_length = self.decoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data["input_ids"].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args["seq_length"] = self.seq_length
            self.megatron_lm_default_args["return_logits"] = self.return_logits
            self.megatron_lm_default_args["tokenizer_type"] = "GPT2BPETokenizer"
        elif "t5" in model.config.model_type.lower():
            model_type_name = "t5"
            num_layers = model.config.num_layers
            hidden_size = model.config.d_model
            num_attention_heads = model.config.num_heads
            max_position_embeddings = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.encoder_seq_length is None:
                if batch_data is not None:
                    self.encoder_seq_length = batch_data["input_ids"].shape[1]
                else:
                    self.encoder_seq_length = max_position_embeddings
            if self.decoder_seq_length is None:
                if batch_data is not None:
                    self.decoder_seq_length = batch_data["labels"].shape[1]
                else:
                    self.decoder_seq_length = max_position_embeddings

            self.megatron_lm_default_args["encoder_seq_length"] = self.encoder_seq_length
            self.megatron_lm_default_args["decoder_seq_length"] = self.decoder_seq_length
        else:
            raise ValueError(
                "🤗 Accelerate Megatron-LM integration supports only BERT, GPT and T5 model. "
                "Please check the model you are using is one of those."
            )

        self.megatron_lm_default_args["model_type_name"] = model_type_name
        self.megatron_lm_default_args["num_layers"] = num_layers
        self.megatron_lm_default_args["hidden_size"] = hidden_size
        self.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
        self.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
        self.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
        self.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
        self.megatron_lm_default_args["model_return_dict"] = model.config.return_dict
        if model_type_name == "bert":
            self.megatron_lm_default_args["num_labels"] = num_labels

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == "fp16":
            self.megatron_lm_default_args["fp16"] = True
        elif mixed_precision == "bf16":
            self.megatron_lm_default_args["bf16"] = True
            self.DDP_impl = "local"
            self.megatron_lm_default_args["DDP_impl"] = self.DDP_impl

    def set_training_args(self, micro_batch_size, dp_degree):
        self.data_parallel_size = dp_degree
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = dp_degree * micro_batch_size * self.num_micro_batches
        self.megatron_lm_default_args["data_parallel_size"] = self.data_parallel_size
        self.megatron_lm_default_args["micro_batch_size"] = self.micro_batch_size
        self.megatron_lm_default_args["global_batch_size"] = self.global_batch_size

    def set_optimizer_type(self, optimizer):
        optimizer_name = optimizer.__class__.__name__.lower()
        if "adam" in optimizer_name:
            self.megatron_lm_default_args["optimizer"] = "adam"
            self.megatron_lm_default_args["adam_beta1"] = optimizer.defaults["betas"][0]
            self.megatron_lm_default_args["adam_beta2"] = optimizer.defaults["betas"][1]
            self.megatron_lm_default_args["adam_eps"] = optimizer.defaults["eps"]
        elif "sgd" in optimizer_name:
            self.megatron_lm_default_args["optimizer"] = "sgd"
            self.megatron_lm_default_args["sgd_momentum"] = optimizer.defaults["momentum"]
        else:
            raise ValueError(f"Optimizer {optimizer_name} is not supported by Megatron-LM")

        self.megatron_lm_default_args["lr"] = optimizer.defaults["lr"]
        self.megatron_lm_default_args["weight_decay"] = optimizer.defaults["weight_decay"]

    def set_scheduler_args(self, scheduler):
        if self.train_iters is None:
            self.train_iters = scheduler.total_num_steps // self.megatron_lm_default_args["data_parallel_size"]
            if self.train_samples is not None:
                self.train_samples = None
                warnings.warn(
                    "Ignoring `train_samples` as `train_iters` based on scheduler is being used for training."
                )
        if self.lr_warmup_iters is None:
            self.lr_warmup_iters = scheduler.warmup_num_steps // self.megatron_lm_default_args["data_parallel_size"]
            if self.lr_warmup_samples is not None:
                warnings.warn(
                    "Ignoring `lr_warmup_samples` as `lr_warmup_iters` based on scheduler is being used for training."
                )
            self.lr_warmup_samples = 0

        self.megatron_lm_default_args["train_iters"] = self.train_iters
        self.megatron_lm_default_args["lr_warmup_iters"] = self.lr_warmup_iters
        self.megatron_lm_default_args["train_samples"] = self.train_samples
        self.megatron_lm_default_args["lr_warmup_samples"] = self.lr_warmup_samples
        self.megatron_lm_default_args["lr_decay_iters"] = self.lr_decay_iters
        self.megatron_lm_default_args["lr_decay_samples"] = self.lr_decay_samples
        self.megatron_lm_default_args["lr_warmup_fraction"] = self.lr_warmup_fraction
        self.megatron_lm_default_args["lr_decay_style"] = self.lr_decay_style
        self.megatron_lm_default_args["weight_decay_incr_style"] = self.weight_decay_incr_style
        self.megatron_lm_default_args["start_weight_decay"] = self.start_weight_decay
        self.megatron_lm_default_args["end_weight_decay"] = self.end_weight_decay
        self.megatron_lm_default_args["min_lr"] = self.min_lr

    def set_tensorboard_logging_options(self):
        from megatron.arguments import _add_logging_args

        parser = argparse.ArgumentParser()
        parser = _add_logging_args(parser)
        logging_args = parser.parse_known_args()
        self.dataset_args = vars(logging_args[0])
        for key, value in self.dataset_args.items():
            if key.startswith("log_"):
                self.megatron_lm_default_args[key] = True
            elif key.startswith("no_log_"):
                self.megatron_lm_default_args[key.replace("no_", "")] = True
