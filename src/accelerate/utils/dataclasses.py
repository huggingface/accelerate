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
import logging
import os
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, get_args

import torch

from .constants import (
    BETA_TP_AVAILABLE_PYTORCH_VERSION,
    BETA_TP_AVAILABLE_TRANSFORMERS_VERSION,
    FSDP2_PYTORCH_VERSION,
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    MITA_PROFILING_AVAILABLE_PYTORCH_VERSION,
    XPU_PROFILING_AVAILABLE_PYTORCH_VERSION,
)
from .environment import parse_flag_from_env, str_to_bool
from .imports import (
    is_cuda_available,
    is_hpu_available,
    is_mlu_available,
    is_msamp_available,
    is_musa_available,
    is_npu_available,
    is_transformer_engine_available,
    is_xpu_available,
)
from .versions import compare_versions, is_torch_version


if TYPE_CHECKING:
    # Mock imports for type checking
    from torchao.float8 import Float8LinearConfig


logger = logging.getLogger(__name__)


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
        # import clear_environment here to avoid circular import problem
        from .environment import clear_environment

        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


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


@dataclass
class AutocastKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how `torch.autocast` behaves. Please refer to the
    documentation of this [context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) for more
    information on each argument.

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import AutocastKwargs

    kwargs = AutocastKwargs(cache_enabled=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    enabled: bool = True
    cache_enabled: bool = None


class DDPCommunicationHookType(BaseEnum):
    """
    Represents a type of communication hook used in DDP.

    Values:

        - **NO** -- no communication hook
        - **FP16** -- DDP communication hook to compress the gradients in FP16
        - **BF16** -- DDP communication hook to compress the gradients in BF16
        - **POWER_SGD** -- DDP communication hook to use PowerSGD
        - **BATCHED_POWER_SGD** -- DDP communication hook to use batched PowerSGD
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    POWER_SGD = "power_sgd"
    BATCHED_POWER_SGD = "batched_power_sgd"


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

    comm_hook: DDPCommunicationHookType = DDPCommunicationHookType.NO
    comm_wrapper: Literal[
        DDPCommunicationHookType.NO,
        DDPCommunicationHookType.FP16,
        DDPCommunicationHookType.BF16,
    ] = DDPCommunicationHookType.NO
    comm_state_option: dict = field(default_factory=dict)

    def to_dict(self, ignore_keys=("comm_hook", "comm_wrapper", "comm_state_option")):
        return {k: v for k, v in super().to_dict().items() if k not in ignore_keys}

    def register_comm_hook(self, model):
        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks,
            powerSGD_hook,
        )

        hook_map: dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_hook,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_hook,
            DDPCommunicationHookType.POWER_SGD: powerSGD_hook.powerSGD_hook,
            DDPCommunicationHookType.BATCHED_POWER_SGD: powerSGD_hook.batched_powerSGD_hook,
        }

        wrapper_map: dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_wrapper,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_wrapper,
        }

        hook: Optional[Callable] = hook_map.get(self.comm_hook)
        wrapper: Optional[Callable] = wrapper_map.get(self.comm_wrapper)

        if hook and wrapper:
            hook = wrapper(hook)

        if hook:
            state = (
                powerSGD_hook.PowerSGDState(None, **self.comm_state_option)
                if self.comm_hook
                in (
                    DDPCommunicationHookType.POWER_SGD,
                    DDPCommunicationHookType.BATCHED_POWER_SGD,
                )
                else None
            )
            model.register_comm_hook(
                state=state,
                hook=hook,
            )


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.amp.GradScaler` or `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `torch.cuda.amp.GradScaler` is only available in PyTorch 1.5.0 and later versions, and `torch.amp.GradScaler` is
    only available in PyTorch 2.4.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs

    kwargs = GradScalerKwargs(backoff_factor=0.25)
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

    Note: If `timeout` is set to `None`, the default will be based upon how `backend` is set.

    ```python
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    backend: Optional[str] = "nccl"
    init_method: Optional[str] = None
    timeout: Optional[timedelta] = None

    def __post_init__(self):
        if self.timeout is None:
            seconds = 1800 if self.backend != "nccl" else 600
            self.timeout = timedelta(seconds=seconds)


# Literals
Backend = Literal["MSAMP", "TE"]
OptLevel = Literal["O1", "O2"]
FP8Format = Literal["HYBRID", "E4M3", "E5M2"]
AmaxComputeAlgorithm = Literal["max", "most_recent"]


# FP8 training recipe kwargs
@dataclass
class AORecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision
    training with `torchao` FP8.

    Args:
        config (`torchao.float8.Float8LinearConfig`, *optional*, default to `None`):
            The configuration for the FP8 training. In general, the default config should be sufficient.
        module_filter_func (`Callable`, *optional*, default to `None`):
            Optional function that must take in a module and layer name, and returns a boolean indicating whether the
            module should be converted to FP8. Defaults to `accelerate.utils.ao.filter_linear_layers`. See it for an
            example.
    """

    config: Optional["Float8LinearConfig"] = None
    module_filter_func: Optional[Callable] = None


@dataclass
class TERecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision
    training with `transformer-engine`.

    <Tip>

        For more information on the args, please refer to the API
        [documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html).

    </Tip>

    ```python
    from accelerate import Accelerator
    from accelerate.utils import TERecipeKwargs

    kwargs = TERecipeKwargs(fp8_format="HYBRID")
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
    ```

    Args:
        use_autocast_during_eval (`bool`, *optional*, default to `False`):
            Whether to use FP8 autocast during eval mode. Generally better metrics are found when this is `False`.
        margin (`int`, *optional*, default to 0):
            The margin to use for the gradient scaling.
        interval (`int`, *optional*, default to 1):
            The interval to use for how often the scaling factor is recomputed.
        fp8_format (`str`, *optional*, default to "HYBRID"):
            The format to use for the FP8 recipe. Must be one of `HYBRID`, `E4M3` or `E5M2`. (Generally `HYBRID` for
            training, `E4M3` or `E5M2` for evaluation)
        amax_history_len (`int`, *optional*, default to 1024):
            The length of the history to use for the scaling factor computation
        amax_compute_algo (`str`, *optional*, default to "most_recent"):
            The algorithm to use for the scaling factor computation. Must be one of `max` or `most_recent`.
        override_linear_precision (`tuple` of three `bool`, *optional*, default to `(False, False, False)`):
            Whether or not to execute `fprop`, `dgrad`, and `wgrad` GEMMS in higher precision.
    """

    use_autocast_during_eval: bool = None
    margin: int = None
    interval: int = None
    fp8_format: FP8Format = None
    amax_history_len: int = None
    amax_compute_algo: AmaxComputeAlgorithm = None
    override_linear_precision: tuple[bool, bool, bool] = None

    def __post_init__(self):
        env_prefix = "ACCELERATE_FP8_"
        if not is_transformer_engine_available():
            raise ImportError("TransformerEngine is not available. Please install it or use a different backend.")
        if self.use_autocast_during_eval is None:
            self.use_autocast_during_eval = parse_flag_from_env(env_prefix + "USE_AUTOCAST_DURING_EVAL")
        if self.margin is None:
            self.margin = int(os.environ.get(env_prefix + "MARGIN", 0))
        if self.interval is None:
            self.interval = int(os.environ.get(env_prefix + "INTERVAL", 1))
        if self.fp8_format is None:
            self.fp8_format = os.environ.get(env_prefix + "FORMAT", "HYBRID")
        self.fp8_format = self.fp8_format.upper()
        if self.fp8_format not in get_args(FP8Format):
            raise ValueError(f"`fp8_format` must be one of {' or '.join(get_args(FP8Format))}.")
        if self.amax_compute_algo is None:
            self.amax_compute_algo = os.environ.get(env_prefix + "AMAX_COMPUTE_ALGO", "most_recent")
        self.amax_compute_algo = self.amax_compute_algo.lower()
        if self.amax_compute_algo not in get_args(AmaxComputeAlgorithm):
            raise ValueError(f"`amax_compute_algo` must be one of {' or '.join(get_args(AmaxComputeAlgorithm))}")
        if self.amax_history_len is None:
            self.amax_history_len = int(os.environ.get(env_prefix + "AMAX_HISTORY_LEN", 1024))
        if self.override_linear_precision is None:
            fprop = parse_flag_from_env(env_prefix + "OVERRIDE_FPROP")
            dgrad = parse_flag_from_env(env_prefix + "OVERRIDE_DGRAD")
            wgrad = parse_flag_from_env(env_prefix + "OVERRIDE_WGRAD")
            self.override_linear_precision = (fprop, dgrad, wgrad)


@dataclass
class MSAMPRecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision
    training with `ms-amp`.
    """

    opt_level: OptLevel = None

    def __post_init__(self):
        env_prefix = "ACCELERATE_FP8_"
        if self.opt_level is None:
            self.opt_level = os.environ.get(env_prefix + "OPT_LEVEL", "O2")
        if self.opt_level not in get_args(OptLevel):
            raise ValueError(f"`opt_level` must be one of {' or '.join(get_args(OptLevel))}")


@dataclass
class FP8RecipeKwargs(TERecipeKwargs, MSAMPRecipeKwargs):
    """
    Deprecated. Please use one of the proper FP8 recipe kwargs classes such as `TERecipeKwargs` or `MSAMPRecipeKwargs`
    instead.
    """

    backend: Backend = None

    def __post_init__(self):
        env_prefix = "ACCELERATE_FP8_"
        warnings.warn(
            "FP8RecipeKwargs is deprecated and will be removed in Accelerate v2.0.0. "
            "Please use one of the proper FP8 recipe kwargs classes such as TERecipeKwargs or MSAMPRecipeKwargs instead.",
            FutureWarning,
        )
        default_backend = "msamp" if is_msamp_available() else "te"
        if self.backend is None:
            self.backend = os.environ.get(env_prefix + "BACKEND", default_backend)
        self.backend = self.backend.upper()
        if self.backend not in get_args(Backend):
            raise ValueError("`backend` must be 'MSAMP' or 'TE' (TransformerEngine) to use `FP8RecipeKwargs`.")
        super().__post_init__()


# Literal
ProfilerActivity = Literal["cpu", "xpu", "mtia", "cuda", "hpu"]


@dataclass
class ProfileKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the profiler. Please refer to the
    documentation of this [context manager](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for
    more information on each argument.

    <Tip warning={true}>

    `torch.profiler` is only available in PyTorch 1.8.1 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import ProfileKwargs

    kwargs = ProfileKwargs(activities=["cpu", "cuda"])
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```

    Args:
        activities (`List[str]`, *optional*, default to `None`):
            The list of activity groups to use in profiling. Must be one of `"cpu"`, `"xpu"`, `"mtia"`, "hpu" or
            `"cuda"`.
        schedule_option (`Dict[str, int]`, *optional*, default to `None`):
            The schedule option to use for the profiler. Available keys are `wait`, `warmup`, `active`, `repeat` and
            `skip_first`. The profiler will skip the first `skip_first` steps, then wait for `wait` steps, then do the
            warmup for the next `warmup` steps, then do the active recording for the next `active` steps and then
            repeat the cycle starting with `wait` steps. The optional number of cycles is specified with the `repeat`
            parameter, the zero value means that the cycles will continue until the profiling is finished.
        on_trace_ready (`Callable`, *optional*, default to `None`):
            Callable that is called at each step when schedule returns `ProfilerAction.RECORD_AND_SAVE` during the
            profiling.
        record_shapes (`bool`, *optional*, default to `False`):
            Save information about operator’s input shapes.
        profile_memory (`bool`, *optional*, default to `False`):
            Track tensor memory allocation/deallocation
        with_stack (`bool`, *optional*, default to `False`):
            Record source information (file and line number) for the ops.
        with_flops (`bool`, *optional*, default to `False`):
            Use formula to estimate the FLOPS of specific operators
        with_modules (`bool`, *optional*, default to `False`):
            Record module hierarchy (including function names) corresponding to the callstack of the op.
        output_trace_dir (`str`, *optional*, default to `None`):
            Exports the collected trace in Chrome JSON format. Chrome use 'chrome://tracing' view json file. Defaults
            to None, which means profiling does not store json files.
    """

    activities: Optional[list[ProfilerActivity]] = None
    schedule_option: Optional[dict[str, int]] = None
    on_trace_ready: Optional[Callable] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    output_trace_dir: Optional[str] = None

    def _get_profiler_activity(self, activity: ProfilerActivity) -> torch.profiler.ProfilerActivity:
        """Get the profiler activity from the string.

        Args:
            activity (str): The profiler activity name.

        Returns:
            torch.profiler.ProfilerActivity: The profiler activity.
        """

        profiler_activity_map: dict[str, torch.profiler.ProfilerActivity] = {
            "cpu": torch.profiler.ProfilerActivity.CPU,
            "cuda": torch.profiler.ProfilerActivity.CUDA,
        }

        if is_hpu_available():
            profiler_activity_map["hpu"] = torch.profiler.ProfilerActivity.HPU

        if is_torch_version(">=", XPU_PROFILING_AVAILABLE_PYTORCH_VERSION):
            if torch.xpu.is_available():
                profiler_activity_map["xpu"] = torch.profiler.ProfilerActivity.XPU

        if is_torch_version(">=", MITA_PROFILING_AVAILABLE_PYTORCH_VERSION):
            if torch.mtia.is_available():
                profiler_activity_map["mtia"] = torch.profiler.ProfilerActivity.MTIA

        if activity not in profiler_activity_map:
            raise ValueError(f"Invalid profiler activity: {activity}. Must be one of {list(profiler_activity_map)}.")
        return profiler_activity_map[activity]

    def build(self) -> torch.profiler.profile:
        """
        Build a profiler object with the current configuration.

        Returns:
            torch.profiler.profile: The profiler object.
        """
        activities: Optional[list[ProfilerActivity]] = None
        if self.activities is not None:
            activities = [self._get_profiler_activity(activity) for activity in self.activities]
        schedule: Optional[torch.profiler.schedule] = None
        if self.schedule_option is not None:
            schedule = torch.profiler.schedule(**self.schedule_option)

        return torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self.on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **MULTI_MLU** -- Distributed on multiple MLUs.
        - **MULTI_SDAA** -- Distributed on multiple SDAAs.
        - **MULTI_MUSA** -- Distributed on multiple MUSAs.
        - **MULTI_NPU** -- Distributed on multiple NPUs.
        - **MULTI_XPU** -- Distributed on multiple XPUs.
        - **MULTI_HPU** -- Distributed on multiple HPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **XLA** -- Using TorchXLA.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    MULTI_NPU = "MULTI_NPU"
    MULTI_MLU = "MULTI_MLU"
    MULTI_SDAA = "MULTI_SDAA"
    MULTI_MUSA = "MULTI_MUSA"
    MULTI_XPU = "MULTI_XPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    XLA = "XLA"
    MEGATRON_LM = "MEGATRON_LM"
    MULTI_HPU = "MULTI_HPU"


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


class FP8BackendType(str, enum.Enum):
    """
    Represents the backend used for FP8.

    Values:

        - **TE** -- using TransformerEngine.
        - **MSAMP** -- using msamp.
    """

    # Subclassing str as well as Enum allows the `FP8BackendType` to be JSON-serializable out of the box.
    NO = "NO"
    TE = "TE"
    MSAMP = "MSAMP"
    AO = "AO"


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


class DynamoBackend(str, BaseEnum):
    """
    Represents a dynamo backend (see https://pytorch.org/docs/stable/torch.compiler.html).

    Values:

        - **NO** -- Do not use torch dynamo.
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **AOT_TS_NVFUSER** -- nvFuser with AotAutograd/TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **NVPRIMS_NVFUSER** -- nvFuser with PrimTorch. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **TENSORRT** -- Uses ONNXRT to run TensorRT for inference optimizations. [Read
          more](https://github.com/onnx/onnx-tensorrt)
        - **AOT_TORCHXLA_TRACE_ONCE** -- Uses Pytorch/XLA with TorchDynamo optimization, for training. [Read
          more](https://github.com/pytorch/xla/blob/r2.0/docs/dynamo.md)
        - **TORCHXLA_TRACE_ONCE** -- Uses Pytorch/XLA with TorchDynamo optimization, for inference. [Read
          more](https://github.com/pytorch/xla/blob/r2.0/docs/dynamo.md)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).
        - **TVM** -- Uses Apach TVM for inference optimizations. [Read more](https://tvm.apache.org/)
        - **HPU_BACKEND** -- Uses HPU backend for inference optimizations.

    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    EAGER = "EAGER"
    AOT_EAGER = "AOT_EAGER"
    INDUCTOR = "INDUCTOR"
    AOT_TS_NVFUSER = "AOT_TS_NVFUSER"
    NVPRIMS_NVFUSER = "NVPRIMS_NVFUSER"
    CUDAGRAPHS = "CUDAGRAPHS"
    OFI = "OFI"
    FX2TRT = "FX2TRT"
    ONNXRT = "ONNXRT"
    TENSORRT = "TENSORRT"
    AOT_TORCHXLA_TRACE_ONCE = "AOT_TORCHXLA_TRACE_ONCE"
    TORCHXLA_TRACE_ONCE = "TORCHXLA_TRACE_ONCE"
    IPEX = "IPEX"
    TVM = "TVM"
    HPU_BACKEND = "HPU_BACKEND"


class LoggerType(BaseEnum):
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **TRACKIO** -- trackio as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
        - **MLFLOW** -- mlflow as an experiment tracker
        - **CLEARML** -- clearml as an experiment tracker
        - **DVCLIVE** -- dvclive as an experiment tracker
        - **SWANLAB** -- swanlab as an experiment tracker
    """

    ALL = "all"
    AIM = "aim"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    TRACKIO = "trackio"
    COMETML = "comet_ml"
    MLFLOW = "mlflow"
    CLEARML = "clearml"
    DVCLIVE = "dvclive"
    SWANLAB = "swanlab"


class PrecisionType(str, BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """

    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class RNGType(BaseEnum):
    TORCH = "torch"
    CUDA = "cuda"
    MLU = "mlu"
    SDAA = "sdaa"
    MUSA = "musa"
    NPU = "npu"
    XLA = "xla"
    XPU = "xpu"
    HPU = "hpu"
    GENERATOR = "generator"


class CustomDtype(enum.Enum):
    r"""
    An enum that contains multiple custom dtypes that can be used for `infer_auto_device_map`.
    """

    FP8 = "fp8"
    INT4 = "int4"
    INT2 = "int2"


# data classes


@dataclass
class TensorInformation:
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class DataLoaderConfiguration:
    """
    Configuration for dataloader-related items when calling `accelerator.prepare`.

    Args:
        split_batches (`bool`, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True`, the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of `num_processes` you are using. If `False`, actual batch size used will be the one set in
            your script multiplied by the number of processes.
        dispatch_batches (`bool`, defaults to `None`):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        use_seedable_sampler (`bool`, defaults to `False`):
            Whether or not use a fully seedable random sampler ([`data_loader.SeedableRandomSampler`]). Ensures
            training results are fully reproducable using a different sampling technique. While seed-to-seed results
            may differ, on average the differences are neglible when using multiple different seeds to compare. Should
            also be ran with [`~utils.set_seed`] for the best results.
        data_seed (`int`, defaults to `None`):
            The seed to use for the underlying generator when using `use_seedable_sampler`. If `None`, the generator
            will use the current default seed from torch.
        non_blocking (`bool`, defaults to `False`):
            If set to `True`, the dataloader prepared by the Accelerator will utilize non-blocking host-to-device
            transfers, allowing for better overlap between dataloader communication and computation. Recommended that
            the prepared dataloader has `pin_memory` set to `True` to work properly.
        use_stateful_dataloader (`bool`, defaults to `False`):
            If set to `True`, the dataloader prepared by the Accelerator will be backed by
            [torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader).
            This requires `torchdata` version 0.8.0 or higher that supports StatefulDataLoader to be installed.
    """

    split_batches: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If"
            " `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a"
            " round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set"
            " in your script multiplied by the number of processes."
        },
    )
    dispatch_batches: bool = field(
        default=None,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process"
            " and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            " underlying dataset is an `IterableDataset`, `False` otherwise."
        },
    )
    even_batches: bool = field(
        default=True,
        metadata={
            "help": "If set to `True`, in cases where the total batch size across all processes does not exactly divide the"
            " dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among"
            " all workers."
        },
    )
    use_seedable_sampler: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use a fully seedable random sampler ([`data_loader.SeedableRandomSampler`])."
            "Ensures training results are fully reproducable using a different sampling technique. "
            "While seed-to-seed results may differ, on average the differences are neglible when using"
            "multiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results."
        },
    )
    data_seed: int = field(
        default=None,
        metadata={
            "help": "The seed to use for the underlying generator when using `use_seedable_sampler`. If `None`, the generator"
            " will use the current default seed from torch."
        },
    )
    non_blocking: bool = field(
        default=False,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator will utilize non-blocking host-to-device"
            " transfers, allowing for better overlap between dataloader communication and computation.  Recommended that the"
            " prepared dataloader has `pin_memory` set to `True` to work properly."
        },
    )
    use_stateful_dataloader: bool = field(
        default=False,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator will be backed by "
            "[torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader). This requires `torchdata` version 0.8.0 or higher that supports StatefulDataLoader to be installed."
        },
    )


@dataclass
class ProjectConfiguration:
    """
    Configuration for the Accelerator object based on inner-project needs.

    Args:
        project_dir (`str`, defaults to `None`):
            A path to a directory for storing data.
        logging_dir (`str`, defaults to `None`):
            A path to a directory for storing logs of locally-compatible loggers. If None, defaults to `project_dir`.
        automatic_checkpoint_naming (`bool`, defaults to `False`):
            Whether saved states should be automatically iteratively named.
        total_limit (`int`, defaults to `None`):
            The maximum number of total saved states to keep.
        iteration (`int`, defaults to `0`):
            The current save iteration.
        save_on_each_node (`bool`, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.
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

    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )

    def set_directories(self, project_dir: str = None):
        "Sets `self.project_dir` and `self.logging_dir` to the appropriate values."
        self.project_dir = project_dir
        if self.logging_dir is None:
            self.logging_dir = project_dir

    def __post_init__(self):
        self.set_directories(self.project_dir)


@dataclass
class GradientAccumulationPlugin(KwargsHandler):
    """
    A plugin to configure gradient accumulation behavior. You can only pass one of `gradient_accumulation_plugin` or
    `gradient_accumulation_steps` to [`Accelerator`]. Passing both raises an error.

    Parameters:
        num_steps (`int`):
            The number of steps to accumulate gradients for.
        adjust_scheduler (`bool`, *optional*, defaults to `True`):
            Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be
            `True` if the used scheduler was not adjusted for gradient accumulation.
        sync_with_dataloader (`bool`, *optional*, defaults to `True`):
            Whether to synchronize setting the gradients when at the end of the dataloader.
        sync_each_batch (`bool`, *optional*):
                Whether to synchronize setting the gradients at each data batch. Seting to `True` may reduce memory
                requirements when using gradient accumulation with distributed training, at expense of speed.

    Example:

    ```python
    from accelerate.utils import GradientAccumulationPlugin

    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2)
    accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin)
    ```
    """

    num_steps: int = field(
        default=None,
        metadata={"help": "The number of steps to accumulate gradients for."},
    )
    adjust_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be `True` if the used scheduler was not adjusted for gradient accumulation."
        },
    )
    sync_with_dataloader: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize setting the gradients when at the end of the dataloader. Should only be set to `False` if you know what you're doing."
        },
    )
    sync_each_batch: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize setting the gradients at each data batch. Setting to `True` may reduce memory requirements when using gradient accumulation with distributed training, at expense of speed."
        },
    )


@dataclass
class TorchDynamoPlugin(KwargsHandler):
    """
    This plugin is used to compile a model with PyTorch 2.0

    Args:
        backend (`DynamoBackend`, defaults to `None`):
            A valid Dynamo backend. See https://pytorch.org/docs/stable/torch.compiler.html for more details.
        mode (`str`, defaults to `None`):
            Possible options are 'default', 'reduce-overhead' or 'max-autotune'.
        fullgraph (`bool`, defaults to `None`):
            Whether it is ok to break model into several subgraphs.
        dynamic (`bool`, defaults to `None`):
            Whether to use dynamic shape for tracing.
        options (`Any`, defaults to `None`):
            A dictionary of options to pass to the backend.
        disable (`bool`, defaults to `False`):
            Turn torch.compile() into a no-op for testing
        use_regional_compilation (`bool`, defaults to `None`):
            Use it to reduce the cold start compilation time of torch.compile() by targeting repeated blocks of the
            same class and compiling them sequentially to hit the compiler's cache. For example, in `GPT2LMHeadModel`,
            the repeated block/class is `GPT2Block`, and can be accessed as `model.transformer.h[0]`. The rest of the
            model (e.g model.lm_head) is compiled separately.
    """

    backend: DynamoBackend = field(
        default=None,
        metadata={"help": f"Possible options are {[b.value.lower() for b in DynamoBackend]}"},
    )
    mode: str = field(
        default=None,
        metadata={"help": "Possible options are 'default', 'reduce-overhead' or 'max-autotune'"},
    )
    fullgraph: bool = field(
        default=None,
        metadata={"help": "Whether it is ok to break model into several subgraphs"},
    )
    dynamic: bool = field(default=None, metadata={"help": "Whether to use dynamic shape for tracing"})
    options: Any = field(
        default=None,
        metadata={"help": "A dictionary of options to pass to the backend."},
    )
    disable: bool = field(
        default=False,
        metadata={"help": "Turn torch.compile() into a no-op for testing"},
    )

    use_regional_compilation: bool = field(
        default=None,
        metadata={
            "help": (
                # https://pytorch.org/tutorials/recipes/regional_compilation.html
                "Use it to reduce the cold start compilation time of torch.compile() by targeting repeated "
                "blocks of the same class and compiling them sequentially to hit the compiler's cache. For "
                "example, in `GPT2LMHeadModel`, the repeated block/class is `GPT2Block`, and can be accessed "
                "as `model.transformer.h[0]`. The rest of the model (e.g model.lm_head) is compiled separately."
            )
        },
    )

    def __post_init__(self):
        prefix = "ACCELERATE_DYNAMO_"
        if self.backend is None:
            self.backend = os.environ.get(prefix + "BACKEND", "no")
        self.backend = DynamoBackend(self.backend.upper())

        if self.mode is None:
            self.mode = os.environ.get(prefix + "MODE", "default")
        if self.fullgraph is None:
            self.fullgraph = str_to_bool(os.environ.get(prefix + "USE_FULLGRAPH", "False")) == 1
        if self.use_regional_compilation is None:
            self.use_regional_compilation = (
                str_to_bool(os.environ.get(prefix + "USE_REGIONAL_COMPILATION", "False")) == 1
            )

        if self.dynamic is None and os.environ.get(prefix + "USE_DYNAMIC", None) is not None:
            self.dynamic = str_to_bool(os.environ.get(prefix + "USE_DYNAMIC", "False")) == 1

    def to_dict(self):
        dynamo_config = copy.deepcopy(self.__dict__)
        dynamo_config["backend"] = dynamo_config["backend"].value.lower()
        return dynamo_config

    def to_kwargs(self):
        kwargs = super().to_kwargs()
        kwargs.pop("use_regional_compilation", None)
        return kwargs


@dataclass
class DeepSpeedPlugin:
    """
    This plugin is used to integrate DeepSpeed.

    Args:
        hf_ds_config (`Any`, defaults to `None`):
            Path to DeepSpeed config file or dict or an object of class `accelerate.utils.deepspeed.HfDeepSpeedConfig`.
        gradient_accumulation_steps (`int`, defaults to `None`):
            Number of steps to accumulate gradients before updating optimizer states. If not set, will use the value
            from the `Accelerator` directly.
        gradient_clipping (`float`, defaults to `None`):
            Enable gradient clipping with value.
        zero_stage (`int`, defaults to `None`):
            Possible options are 0, 1, 2, 3. Default will be taken from environment variable.
        is_train_batch_min (`bool`, defaults to `True`):
            If both train & eval dataloaders are specified, this will decide the `train_batch_size`.
        offload_optimizer_device (`str`, defaults to `None`):
            Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.
        offload_param_device (`str`, defaults to `None`):
            Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.
        offload_optimizer_nvme_path (`str`, defaults to `None`):
            Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.
        offload_param_nvme_path (`str`, defaults to `None`):
            Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.
        zero3_init_flag (`bool`, defaults to `None`):
            Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.
        zero3_save_16bit_model (`bool`, defaults to `None`):
            Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.
        transformer_moe_cls_names (`str`, defaults to `None`):
            Comma-separated list of Transformers MoE layer class names (case-sensitive). For example,
            `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention`, `JetMoEBlock`, etc.
        enable_msamp (`bool`, defaults to `None`):
            Flag to indicate whether to enable MS-AMP backend for FP8 training.
        msasmp_opt_level (`Optional[Literal["O1", "O2"]]`, defaults to `None`):
            Optimization level for MS-AMP (defaults to 'O1'). Only applicable if `enable_msamp` is True. Should be one
            of ['O1' or 'O2'].
    """

    hf_ds_config: Any = field(
        default=None,
        metadata={
            "help": "path to DeepSpeed config file or dict or an object of class `accelerate.utils.deepspeed.HfDeepSpeedConfig`."
        },
    )
    gradient_accumulation_steps: int = field(
        default=None,
        metadata={
            "help": "Number of steps to accumulate gradients before updating optimizer states. If not set, will use the value from the `Accelerator` directly."
        },
    )
    gradient_clipping: float = field(default=None, metadata={"help": "Enable gradient clipping with value"})
    zero_stage: int = field(
        default=None,
        metadata={"help": "Possible options are 0,1,2,3; Default will be taken from environment variable"},
    )
    is_train_batch_min: bool = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"},
    )
    offload_optimizer_device: str = field(
        default=None,
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3."},
    )
    offload_param_device: str = field(
        default=None,
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3."},
    )
    offload_optimizer_nvme_path: str = field(
        default=None,
        metadata={"help": "Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3."},
    )
    offload_param_nvme_path: str = field(
        default=None,
        metadata={"help": "Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3."},
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
    transformer_moe_cls_names: str = field(
        default=None,
        metadata={
            "help": "comma-separated list of transformers MoE layer class names (case-sensitive), e.g : "
            " `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention,JetMoEBlock` ..."
        },
    )
    enable_msamp: bool = field(
        default=None,
        metadata={"help": "Flag to indicate whether to enable MS-AMP backend for FP8 training."},
    )
    msamp_opt_level: Optional[Literal["O1", "O2"]] = field(
        default=None,
        metadata={
            "help": "Optimization level for MS-AMP (defaults to 'O1'). Only applicable if `enable_msamp` is True. Should be one of ['O1' or 'O2']."
        },
    )

    def __post_init__(self):
        from .deepspeed import HfDeepSpeedConfig

        if self.gradient_accumulation_steps is None:
            gas = os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "auto")
            self.gradient_accumulation_steps = int(gas) if gas.isdigit() else gas

        if self.gradient_clipping is None:
            gradient_clipping = os.environ.get("ACCELERATE_GRADIENT_CLIPPING", "auto")
            self.gradient_clipping = gradient_clipping if gradient_clipping == "auto" else float(gradient_clipping)

        if self.zero_stage is None:
            self.zero_stage = int(os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", 2))

        if self.offload_optimizer_device is None:
            self.offload_optimizer_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", "none")

        if self.offload_param_device is None:
            self.offload_param_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE", "none")

        if self.offload_optimizer_nvme_path is None:
            self.offload_optimizer_nvme_path = os.environ.get(
                "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH", "none"
            )

        if self.offload_param_nvme_path is None:
            self.offload_param_nvme_path = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH", "none")

        if self.zero3_save_16bit_model is None:
            self.zero3_save_16bit_model = (
                os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL", "false") == "true"
            )
        if self.enable_msamp is None:
            self.enable_msamp = os.environ.get("ACCELERATE_FP8_BACKEND", None) == "MSAMP"

        if self.msamp_opt_level is None:
            self.msamp_opt_level = os.environ.get("ACCELERATE_FP8_OPT_LEVEL", "O1")

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
            plugin_to_config_mapping = {
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "gradient_clipping": "gradient_clipping",
                "zero_stage": "zero_optimization.stage",
                "offload_optimizer_device": "zero_optimization.offload_optimizer.device",
                "offload_param_device": "zero_optimization.offload_param.device",
                "offload_param_nvme_path": "zero_optimization.offload_param.nvme_path",
                "offload_optimizer_nvme_path": "zero_optimization.offload_optimizer.nvme_path",
                "zero3_save_16bit_model": "zero_optimization.stage3_gather_16bit_weights_on_model_save",
            }
            kwargs = {v: getattr(self, k) for k, v in plugin_to_config_mapping.items() if getattr(self, k) is not None}
            for key in kwargs.keys():
                self.fill_match(key, **kwargs, must_match=False)
            self.hf_ds_config.set_stage_and_offload()

            # filling the missing values in the class attributes from the DeepSpeed config
            # when using the DeepSpeed config file.
            for key, value in plugin_to_config_mapping.items():
                config_value = self.hf_ds_config.get_value(value)
                if config_value is not None and config_value != "auto":
                    setattr(self, key, config_value)
        else:
            config = {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "zero_optimization": {
                    "stage": self.zero_stage,
                    "offload_optimizer": {
                        "device": self.offload_optimizer_device,
                        "nvme_path": (
                            self.offload_optimizer_nvme_path if self.offload_optimizer_device == "nvme" else None
                        ),
                    },
                    "offload_param": {
                        "device": self.offload_param_device,
                        "nvme_path": (self.offload_param_nvme_path if self.offload_param_device == "nvme" else None),
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
                str_to_bool(
                    os.environ.get(
                        "ACCELERATE_DEEPSPEED_ZERO3_INIT",
                        str(self.hf_ds_config.is_zero3()),
                    )
                )
                == 1
            )
        if self.zero3_init_flag and not self.hf_ds_config.is_zero3():
            warnings.warn("DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.")
            self.zero3_init_flag = False
        # NOTE: Set to False by default, will be set to `True` automatically if it's the first plugin passed
        # to the `Accelerator`'s `deepspeed_plugin` param, *or* `AcceleratorState().enable_deepspeed_plugin(plugin_key)` is manually called
        self._set_selected(False)

        # Ignore if it's already set
        if self.enable_msamp and "msamp" not in self.deepspeed_config:
            if self.zero_stage == 3:
                raise NotImplementedError(
                    "MS-AMP is not supported for ZeRO Stage 3. Please use ZeRO Stage 0, 1, or 2 instead."
                )
            if self.msamp_opt_level not in ["O1", "O2"]:
                raise ValueError("Invalid optimization level for MS-AMP. Please use one of ['O1' or'O2'].")
            self.deepspeed_config["msamp"] = {
                "enabled": True,
                "opt_level": self.msamp_opt_level,
            }

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
                    f"Please specify `{ds_key_long}` without `auto` (set to correct value) in the DeepSpeed config file or "
                    "pass it in kwargs."
                )

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_key_long in kwargs:
            if ds_val != kwargs[ds_key_long]:
                mismatches.append(f"- ds {ds_key_long}={ds_val} vs arg {ds_key_long}={kwargs[ds_key_long]}")

    def is_auto(self, ds_key_long):
        val = self.hf_ds_config.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == "auto"

    def get_value(self, ds_key_long, default=None):
        return self.hf_ds_config.get_value(ds_key_long, default)

    def deepspeed_config_process(self, prefix="", mismatches=None, config=None, must_match=True, **kwargs):
        """Process the DeepSpeed config with the values from the kwargs."""
        mismatches = [] if mismatches is None else mismatches
        if config is None:
            config = self.deepspeed_config
        for key, value in config.items():
            if isinstance(value, dict):
                self.deepspeed_config_process(
                    prefix=prefix + key + ".",
                    mismatches=mismatches,
                    config=value,
                    must_match=must_match,
                    **kwargs,
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
            # When training in fp8, we still rely on bf16 autocast for the core mixed precision
            "bf16.enabled": mixed_precision in ("bf16", "fp8"),
        }
        if mixed_precision == "fp16":
            if "fp16" not in ds_config:
                ds_config["fp16"] = {"enabled": True, "auto_cast": True}
        elif mixed_precision in ("bf16", "fp8"):
            if "bf16" not in ds_config:
                ds_config["bf16"] = {"enabled": True}

        if mixed_precision == "fp8" and self.enable_msamp:
            if "msamp" not in ds_config:
                ds_config["msamp"] = {
                    "enabled": True,
                    "opt_level": self.msamp_opt_level,
                }

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

        ds_config = copy.deepcopy(self.deepspeed_config)
        if self.zero3_init_flag:
            if not is_transformers_available():
                raise Exception(
                    "When `zero3_init_flag` is set, it requires Transformers to be installed. "
                    "Please run `pip install transformers`."
                )
        if "gradient_accumulation_steps" not in ds_config or ds_config["gradient_accumulation_steps"] == "auto":
            ds_config["gradient_accumulation_steps"] = 1
        if "train_micro_batch_size_per_gpu" not in ds_config or ds_config["train_micro_batch_size_per_gpu"] == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = 1
        if ds_config.get("train_batch_size", None) == "auto":
            del ds_config["train_batch_size"]

        if compare_versions("transformers", "<", "4.46"):
            from transformers.deepspeed import (
                HfDeepSpeedConfig,
                unset_hf_deepspeed_config,
            )
        else:
            from transformers.integrations import (
                HfDeepSpeedConfig,
                unset_hf_deepspeed_config,
            )

        unset_hf_deepspeed_config()
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
            "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH",
            "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH",
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

    def set_moe_leaf_modules(self, model):
        if self.transformer_moe_cls_names is None:
            self.transformer_moe_cls_names = os.environ.get("ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES", None)
        if self.transformer_moe_cls_names is not None:
            if compare_versions("deepspeed", "<", "0.14.0"):
                raise ImportError("DeepSpeed version must be >= 0.14.0 to use MOE support. Please update DeepSpeed.")
            from deepspeed.utils import set_z3_leaf_modules

            class_names = self.transformer_moe_cls_names.split(",")
            transformer_moe_cls = []
            for layer_class in class_names:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception(
                        f"Could not find a transformer layer class called '{layer_class}' to wrap in the model."
                    )
                else:
                    transformer_moe_cls.append(transformer_cls)
            set_z3_leaf_modules(model, transformer_moe_cls)  # z3_leaf

    def select(self, _from_accelerator_state: bool = False):
        """
        Sets the HfDeepSpeedWeakref to use the current deepspeed plugin configuration
        """
        if not _from_accelerator_state:
            raise ValueError(
                "A `DeepSpeedPlugin` object must be enabled manually by calling `AcceleratorState().enable_deepspeed_plugin(plugin_key)`."
            )
        self.set_deepspeed_weakref()
        self._set_selected(True)

    def _unselect(self):
        self._set_selected(False)

    def _set_selected(self, value: bool):
        """
        Private setter for the 'enabled' attribute.
        """
        self._selected = value

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        raise NotImplementedError(
            "'enabled' can only be set through calling 'AcceleratorState().enable_deepspeed_plugin(key)'."
        )


@dataclass
class FullyShardedDataParallelPlugin:
    """
    This plugin is used to enable fully sharded data parallelism.

    Args:
        fsdp_version (`int`, defaults to `1`):
            The version of FSDP to use. Defaults to 1. If set to 2, launcher expects the config to be converted to
            FSDP2 format.
        sharding_strategy (`Union[str, torch.distributed.fsdp.ShardingStrategy]`, defaults to `'FULL_SHARD'`):
            Sharding strategy to use. Should be either a `str` or an instance of
            `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`. Is deprecated in favor of
            `reshard_after_forward`.
        reshard_after_forward (`Union[str, torch.distributed.fsdp.ShardingStrategy, bool]`, defaults to `'FULL_SHARD'` for `fsdp_version=1` and `True` for `fsdp_version=2`):
            Sharding strategy to use. Should be a bool if `fsdp_version` is set to 2 else a `str` or an instance of
            `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`.
        backward_prefetch (`Union[str, torch.distributed.fsdp.BackwardPrefetch]`, defaults to `'NO_PREFETCH'`):
            Backward prefetch strategy to use. Should be either a `str` or an instance of
            `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`.
        mixed_precision_policy (`Optional[Union[dict, torch.distributed.fsdp.MixedPrecision, torch.distributed.fsdp.MixedPrecisionPolicy]]`, defaults to `None`):
            A config to enable mixed precision training with FullyShardedDataParallel. If passing in a `dict`, it
            should have the following keys: `param_dtype`, `reduce_dtype`, and `buffer_dtype`, can be an instance of
            `torch.distributed.fsdp.MixedPrecisionPolicy` if `fsdp_version` is set to 2.
        auto_wrap_policy (`Optional(Union[Callable, Literal["transformer_based_wrap", "size_based_wrap", "no_wrap"]]), defaults to `NO_WRAP`):
            A callable or string specifying a policy to recursively wrap layers with FSDP. If a string, it must be one
            of `transformer_based_wrap`, `size_based_wrap`, or `no_wrap`. See
            `torch.distributed.fsdp.wrap.size_based_wrap_policy` for a direction on what it should look like.
        cpu_offload (`Union[bool, torch.distributed.fsdp.CPUOffload, torch.distributed.fsdp.CPUOffloadPolicy]`, defaults to `False`):
            Whether to offload parameters to CPU. Should be either a `bool` or an instance of
            `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload` or
            `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffloadPolicy` if `fsdp_version` is set to 2.
        ignored_modules (`Optional[Iterable[torch.nn.Module]]`, defaults to `None`):
            A list of modules to ignore when wrapping with FSDP.
        state_dict_type (`Union[str, torch.distributed.fsdp.StateDictType]`, defaults to `'FULL_STATE_DICT'`):
            State dict type to use. If a string, it must be one of `full_state_dict`, `local_state_dict`, or
            `sharded_state_dict`.
        state_dict_config (`Optional[Union[torch.distributed.fsdp.FullStateDictConfig, torch.distributed.fsdp.ShardedStateDictConfig]`, defaults to `None`):
            State dict config to use. Is determined based on the `state_dict_type` if not passed in.
        optim_state_dict_config (`Optional[Union[torch.distributed.fsdp.FullOptimStateDictConfig, torch.distributed.fsdp.ShardedOptimStateDictConfig]`, defaults to `None`):
            Optim state dict config to use. Is determined based on the `state_dict_type` if not passed in.
        limit_all_gathers (`bool`, defaults to `True`):
            Whether to have FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers. This
            bool only affects the sharded strategies that schedule all-gathers. Enabling this can help lower the number
            of CUDA malloc retries.
        use_orig_params (`bool`, defaults to `False`):
            Whether to use the original parameters for the optimizer.
        param_init_fn (`Optional[Callable[[torch.nn.Module], None]`, defaults to `None`):
            A `Callable[torch.nn.Module] -> None` that specifies how modules that are currently on the meta device
            should be initialized onto an actual device. Only applicable when `sync_module_states` is `True`. By
            default is a `lambda` which calls `to_empty` on the module.
        sync_module_states (`bool`, defaults to `False`):
            Whether each individually wrapped FSDP unit should broadcast module parameters from rank 0 to ensure they
            are the same across all ranks after initialization. Defaults to `False` unless `cpu_ram_efficient_loading`
            is `True`, then will be forcibly enabled.
        forward_prefetch (`bool`, defaults to `False`):
            Whether to have FSDP explicitly prefetches the next upcoming all-gather while executing in the forward
            pass. only use with Static graphs.
        activation_checkpointing (`bool`, defaults to `False`):
            A technique to reduce memory usage by clearing activations of certain layers and recomputing them during a
            backward pass. Effectively, this trades extra computation time for reduced memory usage.
        cpu_ram_efficient_loading (`bool`, defaults to `None`):
            If True, only the first process loads the pretrained model checkoint while all other processes have empty
            weights. Only applicable for Transformers. When using this, `sync_module_states` needs to be `True`.
        transformer_cls_names_to_wrap (`Optional[List[str]]`, defaults to `None`):
            A list of transformer layer class names to wrap. Only applicable when `auto_wrap_policy` is
            `transformer_based_wrap`.
        min_num_params (`Optional[int]`, defaults to `None`):
            The minimum number of parameters a module must have to be wrapped. Only applicable when `auto_wrap_policy`
            is `size_based_wrap`.
    """

    fsdp_version: int = field(
        default=None,
        metadata={
            "help": "The version of FSDP to use. Defaults to 1. If set to 2, launcher expects the config to be converted to FSDP2 format."
        },
    )

    sharding_strategy: Union[str, "torch.distributed.fsdp.ShardingStrategy"] = field(
        default=None,
        metadata={
            "help": "Sharding strategy to use. Should be either a `str` or an instance of `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`. Defaults to 'FULL_SHARD'. Is deprecated in favor of `reshard_after_forward` "
        },
    )

    reshard_after_forward: Union[str, "torch.distributed.fsdp.ShardingStrategy", bool] = field(
        default=None,
        metadata={
            "help": "Sharding strategy to use. Should be a bool if `fsdp_version` is set to 2 else a `str` or an instance of `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`. Defaults to 'FULL_SHARD'"
        },
    )
    backward_prefetch: Optional[Union[str, "torch.distributed.fsdp.BackwardPrefetch"]] = field(
        default=None,
        metadata={
            "help": "Backward prefetch strategy to use. Should be either a `str` or an instance of `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`. Defaults to 'NO_PREFETCH'. This becomes obsolete in FSDP2."
        },
    )
    mixed_precision_policy: Optional[
        Union[
            dict,
            "torch.distributed.fsdp.MixedPrecision",
            "torch.distributed.fsdp.MixedPrecisionPolicy",
        ]
    ] = field(
        default=None,
        metadata={
            "help": "A config to enable mixed precision training with FullyShardedDataParallel. "
            "If passing in a `dict`, it should have the following keys: `param_dtype`, `reduce_dtype`, and `buffer_dtype`."
            "Can also be an instance of `torch.distributed.fsdp.MixedPrecisionPolicy` if `fsdp_version` is set to 2."
        },
    )
    auto_wrap_policy: Optional[Union[Callable, Literal["transformer_based_wrap", "size_based_wrap", "no_wrap"]]] = (
        field(
            default=None,
            metadata={
                "help": "A callable or string specifying a policy to recursively wrap layers with FSDP. If a string, it must be one of `transformer_based_wrap`, `size_based_wrap`, or `no_wrap`. "
                "Defaults to `NO_WRAP`. See `torch.distributed.fsdp.wrap.size_based_wrap_policy` for a direction on what it should look like"
            },
        )
    )
    cpu_offload: Union[
        bool,
        "torch.distributed.fsdp.CPUOffload",
        "torch.distributed.fsdp.CPUOffloadPolicy",
    ] = field(
        default=None,
        metadata={
            "help": "Whether to offload parameters to CPU. Should be either a `bool` or an instance of `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload` or `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffloadPolicy` if `fsdp_version` is set to 2. Defaults to `False`"
        },
    )
    ignored_modules: Optional[Iterable[torch.nn.Module]] = field(
        default=None,
        metadata={"help": "A list of modules to ignore when wrapping with FSDP."},
    )

    state_dict_type: Union[str, "torch.distributed.fsdp.StateDictType"] = field(
        default=None,
        metadata={
            "help": "State dict type to use. If a string, it must be one of `full_state_dict`, `local_state_dict`, or `sharded_state_dict`. Defaults to `FULL_STATE_DICT`"
        },
    )
    state_dict_config: Optional[
        Union[
            "torch.distributed.fsdp.FullStateDictConfig",
            "torch.distributed.fsdp.ShardedStateDictConfig",
        ]
    ] = field(
        default=None,
        metadata={"help": "State dict config to use. Is determined based on the `state_dict_type` if not passed in."},
    )
    optim_state_dict_config: Optional[
        Union[
            "torch.distributed.fsdp.FullOptimStateDictConfig",
            "torch.distributed.fsdp.ShardedOptimStateDictConfig",
        ]
    ] = field(
        default=None,
        metadata={
            "help": "Optim state dict config to use. Is determined based on the `state_dict_type` if not passed in."
        },
    )
    limit_all_gathers: bool = field(
        default=True,
        metadata={
            "help": "Whether to have FSDP explicitly synchronizes the CPU thread to prevent "
            "too many in-flight all-gathers. This bool only affects the sharded strategies that schedule all-gathers. "
            "Enabling this can help lower the number of CUDA malloc retries."
        },
    )
    use_orig_params: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use the original parameters for the optimizer. Defaults to `False`. This becomes obsolete in FSDP2."
        },
    )
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = field(
        default=None,
        metadata={
            "help": "A Callable[torch.nn.Module] -> None that specifies how modules "
            "that are currently on the meta device should be initialized onto an actual device. "
            "Only applicable when `sync_module_states` is `True`. By default is a `lambda` which calls `to_empty` on the module."
        },
    )
    sync_module_states: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether each individually wrapped FSDP unit should broadcast module parameters from rank 0 "
            "to ensure they are the same across all ranks after initialization. Defaults to `False` unless "
            "`cpu_ram_efficient_loading` is `True`, then will be forcibly enabled. This becomes obsolete in FSDP2."
        },
    )
    forward_prefetch: bool = field(
        default=None,
        metadata={
            "help": "Whether to have FSDP explicitly prefetches the next upcoming "
            "all-gather while executing in the forward pass. only use with Static graphs. Defaults to `False`"
        },
    )
    activation_checkpointing: bool = field(
        default=None,
        metadata={
            "help": "A technique to reduce memory usage by clearing activations of "
            "certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time "
            "for reduced memory usage. Defaults to `False`"
        },
    )
    cpu_ram_efficient_loading: bool = field(
        default=None,
        metadata={
            "help": "If True, only the first process loads the pretrained model checkoint while all other processes have empty weights. "
            "Only applicable for 🤗 Transformers. When using this, `sync_module_states` needs to be `True`. Defaults to `False`."
        },
    )
    transformer_cls_names_to_wrap: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "A list of transformer layer class names to wrap. Only applicable when `auto_wrap_policy` is `transformer_based_wrap`."
        },
    )
    min_num_params: Optional[int] = field(
        default=None,
        metadata={
            "help": "The minimum number of parameters a module must have to be wrapped. Only applicable when `auto_wrap_policy` is `size_based_wrap`."
        },
    )

    def __post_init__(self):
        from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy

        _fsdp2_warnings = set()

        env_prefix = "FSDP_"
        # Strategy: By default we should always assume that values are passed in, else we check the environment variables
        if self.fsdp_version is None:
            self.fsdp_version = int(os.environ.get(env_prefix + "VERSION", "1"))

        if self.fsdp_version == 2:
            if not is_torch_version(">=", FSDP2_PYTORCH_VERSION):
                raise ImportError(f"FSDP2 requires PyTorch >= {FSDP2_PYTORCH_VERSION}")

        if self.sharding_strategy is not None:
            # We cannot properly detect all of the cases, as by default `args.fsdp_sharding_strategy` is set to `fully_shard`
            # Therefore we issue a warning only if the user has explicitly set it inside their plugin
            _fsdp2_warnings.add(
                "sharding_strategy is deprecated in favor of reshard_after_forward. "
                "This will be removed in a future version of Accelerate."
            )
        if self.fsdp_version == 1:
            if self.sharding_strategy is None:
                self.sharding_strategy = os.environ.get(env_prefix + "SHARDING_STRATEGY", "FULL_SHARD")
            if isinstance(self.sharding_strategy, str):
                if self.sharding_strategy.upper() in FSDP_SHARDING_STRATEGY:
                    self.sharding_strategy = FSDP_SHARDING_STRATEGY.index(self.sharding_strategy.upper()) + 1
                if isinstance(self.sharding_strategy, int) or self.sharding_strategy.isdigit():
                    self.sharding_strategy = ShardingStrategy(int(self.sharding_strategy))
                else:
                    self.sharding_strategy = ShardingStrategy[self.sharding_strategy.upper()]

        # Fallback to `reshard_after_forward` in FSDP1 if `sharding_strategy` is not set
        if self.reshard_after_forward is None and self.sharding_strategy is None:
            reshard_after_forward = os.environ.get(
                env_prefix + "RESHARD_AFTER_FORWARD",
                "true" if self.fsdp_version == 2 else "FULL_SHARD",
            )
            if self.fsdp_version == 2:
                self.reshard_after_forward = str_to_bool(reshard_after_forward.lower(), to_bool=True)
            else:
                self.reshard_after_forward = reshard_after_forward
        if isinstance(self.reshard_after_forward, str):
            if self.fsdp_version == 2:
                self.reshard_after_forward = str_to_bool(self.reshard_after_forward.lower(), to_bool=True)
            else:
                # We need to remap based on custom enum values for user readability
                if self.reshard_after_forward.upper() in FSDP_SHARDING_STRATEGY:
                    self.reshard_after_forward = FSDP_SHARDING_STRATEGY.index(self.reshard_after_forward.upper()) + 1
                if isinstance(self.reshard_after_forward, int) or self.reshard_after_forward.isdigit():
                    self.reshard_after_forward = ShardingStrategy(int(self.reshard_after_forward))
                else:
                    self.reshard_after_forward = ShardingStrategy[self.reshard_after_forward.upper()]

        if self.fsdp_version == 2 and not isinstance(self.reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward set to {self.reshard_after_forward}. This is not supported with FSDP2, please set to a `bool`"
            )
        if self.fsdp_version == 1 and isinstance(self.reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward set to {self.reshard_after_forward}. This is not supported with FSDP1, please set to a `str` or an instance of `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`"
            )

        if self.cpu_offload is None:
            self.cpu_offload = str_to_bool(os.environ.get(env_prefix + "OFFLOAD_PARAMS", "False")) == 1

        self.set_cpu_offload()  # abstracted away to hide imports due to version checks
        self.validate_cpu_offload()

        if self.backward_prefetch is None:
            self.backward_prefetch = os.environ.get(env_prefix + "BACKWARD_PREFETCH", None)
        if isinstance(self.backward_prefetch, str) and self.backward_prefetch.upper() == "NO_PREFETCH":
            self.backward_prefetch = None
        if self.backward_prefetch is not None and not isinstance(self.backward_prefetch, BackwardPrefetch):
            if isinstance(self.backward_prefetch, str) and self.backward_prefetch.upper() in FSDP_BACKWARD_PREFETCH:
                self.backward_prefetch = FSDP_BACKWARD_PREFETCH.index(self.backward_prefetch.upper()) + 1
            if isinstance(self.backward_prefetch, int) or self.backward_prefetch.isdigit():
                self.backward_prefetch = BackwardPrefetch(int(self.backward_prefetch))
            else:
                self.backward_prefetch = BackwardPrefetch[self.backward_prefetch.upper()]
        if self.fsdp_version == 2 and self.backward_prefetch is not None:
            _fsdp2_warnings.add("backward_prefetch is not supported in FSDP2. Setting backward prefetch to None.")
            self.backward_prefetch = None

        self.set_state_dict_type()

        if self.auto_wrap_policy is None:
            self.auto_wrap_policy = os.environ.get(env_prefix + "AUTO_WRAP_POLICY", "NO_WRAP")
        if isinstance(self.auto_wrap_policy, str):
            if self.auto_wrap_policy.upper() not in FSDP_AUTO_WRAP_POLICY:
                raise ValueError(
                    f"Invalid auto wrap policy: {self.auto_wrap_policy}. Must be one of {FSDP_AUTO_WRAP_POLICY}"
                )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            if self.auto_wrap_policy.upper() == "TRANSFORMER_BASED_WRAP":
                self.auto_wrap_policy = transformer_auto_wrap_policy
                if self.transformer_cls_names_to_wrap is None:
                    self.transformer_cls_names_to_wrap = os.environ.get(env_prefix + "TRANSFORMER_CLS_TO_WRAP", None)
                if isinstance(self.transformer_cls_names_to_wrap, str):
                    self.transformer_cls_names_to_wrap = self.transformer_cls_names_to_wrap.split(",")
            elif self.auto_wrap_policy.upper() == "SIZE_BASED_WRAP":
                self.auto_wrap_policy = size_based_auto_wrap_policy
                if self.min_num_params is None:
                    self.min_num_params = int(os.environ.get(env_prefix + "MIN_NUM_PARAMS", 0))
                elif not isinstance(self.min_num_params, int):
                    raise ValueError(
                        f"`min_num_params` must be an integer. Got {self.min_num_params} of type {type(self.min_num_params)}"
                    )
            elif self.auto_wrap_policy.upper() == "NO_WRAP":
                self.auto_wrap_policy = None

        if self.use_orig_params is None and self.fsdp_version == 1:
            self.use_orig_params = str_to_bool(os.environ.get(env_prefix + "USE_ORIG_PARAMS", "False")) == 1
        if self.fsdp_version == 2 and self.use_orig_params is not None:
            _fsdp2_warnings.add("use_orig_params is obsolete in FSDP2, as FSDP2 always uses the original parameters.")
            self.use_orig_params = None

        if self.sync_module_states is None and self.fsdp_version == 1:
            self.sync_module_states = str_to_bool(os.environ.get(env_prefix + "SYNC_MODULE_STATES", "False")) == 1
        if self.fsdp_version == 2 and self.sync_module_states is not None:
            _fsdp2_warnings.add(
                "sync_module_states is obsolete in FSDP2, as it is not needed anymore."
                "Setting sync_module_states to None."
            )
            self.sync_module_states = None

        if self.forward_prefetch is None and self.fsdp_version == 1:
            self.forward_prefetch = str_to_bool(os.environ.get(env_prefix + "FORWARD_PREFETCH", "False")) == 1
        if self.fsdp_version == 2 and self.forward_prefetch is not None:
            raise ValueError("forward_prefetch is not yet implemented in FSDP2, set to None or use `fsdp_version=1`")

        if self.activation_checkpointing is None:
            self.activation_checkpointing = (
                str_to_bool(os.environ.get(env_prefix + "ACTIVATION_CHECKPOINTING", "False")) == 1
            )

        if self.cpu_ram_efficient_loading is None:
            self.cpu_ram_efficient_loading = (
                str_to_bool(os.environ.get(env_prefix + "CPU_RAM_EFFICIENT_LOADING", "False")) == 1
            )
        # There's no need to specify sync_module_states in FSDP2
        if self.fsdp_version == 1 and self.cpu_ram_efficient_loading and not self.sync_module_states:
            warnings.warn(
                "sync_module_states cannot be False since efficient cpu ram loading enabled. "
                "Setting sync_module_states to True."
            )
            self.sync_module_states = True

        if self.cpu_ram_efficient_loading != bool(
            str_to_bool(os.environ.get(env_prefix + "CPU_RAM_EFFICIENT_LOADING", "False"))
        ):
            env_var = env_prefix + "CPU_RAM_EFFICIENT_LOADING"
            warnings.warn(
                f"The `cpu_ram_efficient_loading` flag for `FullyShardedDataParallelPlugin` does not match the environment variable {env_var}. "
                "Setting environment variable to match `cpu_ram_efficient_loading`."
            )
            os.environ[env_var] = str(self.cpu_ram_efficient_loading)

        if isinstance(self.mixed_precision_policy, dict):
            self.set_mixed_precision(self.mixed_precision_policy)
        if self.mixed_precision_policy is not None:
            self.validate_mixed_precision_policy()

        if self.sync_module_states:
            if is_npu_available():
                device = torch.npu.current_device()
            elif is_mlu_available():
                device = torch.mlu.current_device()
            elif is_musa_available():
                device = torch.musa.current_device()
            elif is_cuda_available():
                device = torch.cuda.current_device()
            elif is_xpu_available():
                device = torch.xpu.current_device()
            elif is_hpu_available():
                device = torch.hpu.current_device()
            else:
                raise RuntimeError(
                    "There are currently no available devices found, must be one of 'XPU', 'CUDA', 'MLU', 'NPU', 'MUSA', or 'HPU'."
                )
            # Create a function that will be used to initialize the parameters of the model
            # when using `sync_module_states`
            self.param_init_fn = lambda x: x.to_empty(device=device, recurse=False)

        #  Single warning for all deprecation warnings due to FSDP2 conversion
        if _fsdp2_warnings:
            logger.warning("Multiple deprecation warnings due to FSDP2 conversion:\n".join(_fsdp2_warnings))

    def set_state_dict_type(self, state_dict_type=None):
        """
        Set the state dict config based on the `StateDictType`.
        """
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        # Override the state_dict_type if provided, typical use case:
        # user trains with sharded, but final save is with full
        if state_dict_type is not None:
            self.state_dict_type = state_dict_type

        if self.state_dict_type is None:
            self.state_dict_type = os.environ.get(
                "FSDP_STATE_DICT_TYPE",
                "FULL_STATE_DICT" if self.fsdp_version == 1 else "SHARDED_STATE_DICT",
            )
        if isinstance(self.state_dict_type, str):
            if self.state_dict_type.isdigit():
                self.state_dict_type = StateDictType(int(self.state_dict_type))
            else:
                self.state_dict_type = StateDictType[self.state_dict_type.upper()]

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            if self.state_dict_config is None:
                self.state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            if self.optim_state_dict_config is None:
                self.optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        elif self.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            if self.state_dict_config is None:
                self.state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
            if self.optim_state_dict_config is None:
                self.optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)

        if self.fsdp_version == 2 and self.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise ValueError(
                "FSDP2 does not support LOCAL_STATE_DICT. "
                "Please set `fsdp_state_dict_type` to `SHARDED_STATE_DICT` or `FULL_STATE_DICT`."
            )

    def set_auto_wrap_policy(self, model):
        """
        Given `model`, creates an `auto_wrap_policy` baesd on the passed in policy and if we can use the
        `transformer_cls_to_wrap`
        """
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        # First base off of `_no_split_modules`
        no_split_modules = getattr(model, "_no_split_modules", None)
        default_transformer_cls_names_to_wrap = list(no_split_modules) if no_split_modules is not None else []
        if self.auto_wrap_policy == transformer_auto_wrap_policy:
            if self.transformer_cls_names_to_wrap is None:
                self.transformer_cls_names_to_wrap = default_transformer_cls_names_to_wrap
            transformer_cls_to_wrap = set()
            for layer_class in self.transformer_cls_names_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
                transformer_cls_to_wrap.add(transformer_cls)
            # Finally we set the auto_wrap_policy to a callable
            self.auto_wrap_policy = functools.partial(
                self.auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap
            )

        elif self.auto_wrap_policy == size_based_auto_wrap_policy:
            # If zero, we silently ignore it.
            if self.min_num_params > 0:
                self.auto_wrap_policy = functools.partial(self.auto_wrap_policy, min_num_params=self.min_num_params)
            else:
                self.auto_wrap_policy = None

    def set_mixed_precision(self, mixed_precision, buffer_autocast=False, override=False):
        "Sets the mixed precision policy for FSDP"
        mixed_precision_mapping = {
            "fp8": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = mixed_precision
        if isinstance(mixed_precision, str):
            dtype = mixed_precision_mapping.get(mixed_precision, None)
            if dtype is None:
                raise ValueError(
                    f"Invalid mixed precision: {mixed_precision}. Must be one of {list(mixed_precision_mapping.keys())}"
                )
        elif isinstance(mixed_precision, torch.dtype) and mixed_precision not in mixed_precision_mapping.values():
            raise ValueError(
                f"Invalid mixed precision: {mixed_precision}. Must be one of {list(mixed_precision_mapping.values())}"
            )

        buffer_type = torch.float32 if buffer_autocast else dtype

        if self.fsdp_version == 1:
            from torch.distributed.fsdp import MixedPrecision
        elif self.fsdp_version == 2:
            from torch.distributed.fsdp import MixedPrecisionPolicy as MixedPrecision

        if override or self.mixed_precision_policy is None:
            dtype_args = {"param_dtype": dtype, "reduce_dtype": dtype}
            if self.fsdp_version == 1:
                dtype_args["buffer_dtype"] = buffer_type
            else:
                dtype_args["output_dtype"] = dtype
            # TODO(s1ro1): `cast_forward_inputs` for FSDP2?
            self.mixed_precision_policy = MixedPrecision(**dtype_args)
        elif isinstance(self.mixed_precision_policy, dict):
            # Check for incompatible types
            valid_keys = ["param_dtype", "reduce_dtype"] + (
                ["buffer_dtype"] if self.fsdp_version == 1 else ["output_dtype"]
            )
            missing_keys = [k for k in valid_keys if k not in self.mixed_precision_policy]
            invalid_values = [
                k for k, v in self.mixed_precision_policy.items() if v not in mixed_precision_mapping.values()
            ]
            if missing_keys or invalid_values:
                raise ValueError(
                    f"Invalid mixed precision policy: {self.mixed_precision_policy}. "
                    f"Must be a `dict` with keys {valid_keys}."
                    f"Values must be one of {list(mixed_precision_mapping.values())}"
                )
            self.mixed_precision_policy = MixedPrecision(**self.mixed_precision_policy)

    def validate_mixed_precision_policy(self):
        """
        Validates the mixed precision policy, abstracted away to not bring in the imports if not needed.
        """
        if self.fsdp_version == 2:
            from torch.distributed.fsdp import MixedPrecisionPolicy as MixedPrecision
        else:
            from torch.distributed.fsdp import MixedPrecision

        if not isinstance(self.mixed_precision_policy, MixedPrecision):
            required_type = (
                "`torch.distributed.fsdp.MixedPrecisionPolicy`"
                if self.fsdp_version == 2
                else "`torch.distributed.fsdp.MixedPrecision`"
            )
            raise ValueError(f"mixed_precision_policy must be an instance of {required_type}.")

    def set_cpu_offload(self):
        if self.fsdp_version == 2:
            from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy
        else:
            from torch.distributed.fsdp import CPUOffload

        if isinstance(self.cpu_offload, bool):
            if self.fsdp_version == 2:
                if not self.cpu_offload:
                    self.cpu_offload = OffloadPolicy()
                else:
                    self.cpu_offload = CPUOffloadPolicy()
            else:
                self.cpu_offload = CPUOffload(offload_params=self.cpu_offload)

    def validate_cpu_offload(self):
        if self.fsdp_version == 2:
            from torch.distributed.fsdp import OffloadPolicy
        else:
            from torch.distributed.fsdp import CPUOffload

        if self.fsdp_version == 2 and not isinstance(self.cpu_offload, OffloadPolicy):
            raise ValueError(
                f"`cpu_offload` must be an instance of `torch.distributed.fsdp.OffloadPolicy` in FSDP2, got {self.cpu_offload}"
            )
        if self.fsdp_version == 1 and not isinstance(self.cpu_offload, CPUOffload):
            raise ValueError(
                f"`cpu_offload` must be an instance of `torch.distributed.fsdp.CPUOffload` in FSDP1, got {self.cpu_offload}"
            )


@dataclass
class TorchTensorParallelPlugin:
    """
    This plugin is used to enable tensor parallelism using PyTorch >= 2.0.
    """

    tp_size: int = field(
        default=1,
        metadata={"help": "tensor parallel size will be used in the device mesh preparation"},
    )

    # torch_device_mesh is of type "torch.distributed.DeviceMesh"
    torch_device_mesh: Optional["torch.distributed.DeviceMesh"] = field(default=None)


@dataclass
class TorchTensorParallelConfig:
    """
    Use this object in your [`Accelerator`] to customize your torch tensor parallelism.
    """

    enable_async_tp: bool = False

    def __post_init__(self):
        if not is_torch_version(">=", BETA_TP_AVAILABLE_PYTORCH_VERSION):
            raise ValueError(
                f"Torch tensor parallelism is only available in PyTorch {BETA_TP_AVAILABLE_PYTORCH_VERSION} and later versions. "
                "Please upgrade your PyTorch version."
            )

        if not compare_versions("transformers", ">=", BETA_TP_AVAILABLE_TRANSFORMERS_VERSION):
            raise ValueError(f"TP requires transformers >= {BETA_TP_AVAILABLE_TRANSFORMERS_VERSION}")

        if self.enable_async_tp:
            warnings.warn("Async tensor parallelism is currently not supported, ignoring this option.")


@dataclass
class MegatronLMPlugin:
    """
    Plugin for Megatron-LM to enable tensor, pipeline, sequence and data parallelism. Also to enable selective
    activation recomputation and optimized fused kernels.

    Args:
        tp_degree (`int`, defaults to `None`):
            Tensor parallelism degree.
        pp_degree (`int`, defaults to `None`):
            Pipeline parallelism degree.
        num_micro_batches (`int`, defaults to `None`):
            Number of micro-batches.
        gradient_clipping (`float`, defaults to `None`):
            Gradient clipping value based on global L2 Norm (0 to disable).
        sequence_parallelism (`bool`, defaults to `None`):
            Enable sequence parallelism.
        recompute_activations (`bool`, defaults to `None`):
            Enable selective activation recomputation.
        use_distributed_optimizr (`bool`, defaults to `None`):
            Enable distributed optimizer.
        pipeline_model_parallel_split_rank (`int`, defaults to `None`):
            Rank where encoder and decoder should be split.
        num_layers_per_virtual_pipeline_stage (`int`, defaults to `None`):
            Number of layers per virtual pipeline stage.
        is_train_batch_min (`str`, defaults to `True`):
            If both tran & eval dataloaders are specified, this will decide the `micro_batch_size`.
        train_iters (`int`, defaults to `None`):
            Total number of samples to train over all training runs. Note that either train-iters or train-samples
            should be provided when using `MegatronLMDummyScheduler`.
        train_samples (`int`, defaults to `None`):
            Total number of samples to train over all training runs. Note that either train-iters or train-samples
            should be provided when using `MegatronLMDummyScheduler`.
        weight_decay_incr_style (`str`, defaults to `'constant'`):
            Weight decay increment function. choices=["constant", "linear", "cosine"].
        start_weight_decay (`float`, defaults to `None`):
            Initial weight decay coefficient for L2 regularization.
        end_weight_decay (`float`, defaults to `None`):
            End of run weight decay coefficient for L2 regularization.
        lr_decay_style (`str`, defaults to `'linear'`):
            Learning rate decay function. choices=['constant', 'linear', 'cosine'].
        lr_decay_iters (`int`, defaults to `None`):
            Number of iterations for learning rate decay. If None defaults to `train_iters`.
        lr_decay_samples (`int`, defaults to `None`):
            Number of samples for learning rate decay. If None defaults to `train_samples`.
        lr_warmup_iters (`int`, defaults to `None`):
            Number of iterations to linearly warmup learning rate over.
        lr_warmup_samples (`int`, defaults to `None`):
            Number of samples to linearly warmup learning rate over.
        lr_warmup_fraction (`float`, defaults to `None`):
            Fraction of lr-warmup-(iters/samples) to linearly warmup learning rate over.
        min_lr (`float`, defaults to `0`):
            Minumum value for learning rate. The scheduler clip values below this threshold.
        consumed_samples (`List`, defaults to `None`):
            Number of samples consumed in the same order as the dataloaders to `accelerator.prepare` call.
        no_wd_decay_cond (`Optional`, defaults to `None`):
            Condition to disable weight decay.
        scale_lr_cond (`Optional`, defaults to `None`):
            Condition to scale learning rate.
        lr_mult (`float`, defaults to `1.0`):
            Learning rate multiplier.
        megatron_dataset_flag (`bool`, defaults to `False`):
            Whether the format of dataset follows Megatron-LM Indexed/Cached/MemoryMapped format.
        seq_length (`int`, defaults to `None`):
            Maximum sequence length to process.
        encoder_seq_length (`int`, defaults to `None`):
            Maximum sequence length to process for the encoder.
        decoder_seq_length (`int`, defaults to `None`):
            Maximum sequence length to process for the decoder.
        tensorboard_dir (`str`, defaults to `None`):
            Path to save tensorboard logs.
        set_all_logging_options (`bool`, defaults to `False`):
            Whether to set all logging options.
        eval_iters (`int`, defaults to `100`):
            Number of iterations to run for evaluation validation/test for.
        eval_interval (`int`, defaults to `1000`):
            Interval between running evaluation on validation set.
        return_logits (`bool`, defaults to `False`):
            Whether to return logits from the model.
        custom_train_step_class (`Optional`, defaults to `None`):
            Custom train step class.
        custom_train_step_kwargs (`Optional`, defaults to `None`):
            Custom train step kwargs.
        custom_model_provider_function (`Optional`, defaults to `None`):
            Custom model provider function.
        custom_prepare_model_function (`Optional`, defaults to `None`):
            Custom prepare model function.
        custom_megatron_datasets_provider_function (`Optional`, defaults to `None`):
            Custom megatron train_valid_test datasets provider function.
        custom_get_batch_function (`Optional`, defaults to `None`):
            Custom get batch function.
        custom_loss_function (`Optional`, defaults to `None`):
            Custom loss function.
        other_megatron_args (`Optional`, defaults to `None`):
            Other Megatron-LM arguments. Please refer Megatron-LM.
    """

    tp_degree: int = field(default=None, metadata={"help": "tensor parallelism degree."})
    pp_degree: int = field(default=None, metadata={"help": "pipeline parallelism degree."})
    num_micro_batches: int = field(default=None, metadata={"help": "number of micro-batches."})
    gradient_clipping: float = field(
        default=None,
        metadata={"help": "gradient clipping value based on global L2 Norm (0 to disable)"},
    )
    sequence_parallelism: bool = field(
        default=None,
        metadata={"help": "enable sequence parallelism"},
    )
    recompute_activations: bool = field(
        default=None,
        metadata={"help": "enable selective activation recomputation"},
    )
    use_distributed_optimizer: bool = field(
        default=None,
        metadata={"help": "enable distributed optimizer"},
    )
    pipeline_model_parallel_split_rank: int = field(
        default=None,
        metadata={"help": "Rank where encoder and decoder should be split."},
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
    consumed_samples: list[int] = field(
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
        default=100,
        metadata={"help": "Number of iterations to run for evaluation validation/test for."},
    )
    eval_interval: int = field(
        default=1000,
        metadata={"help": "Interval between running evaluation on validation set."},
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
    custom_train_step_kwargs: Optional[dict[str, Any]] = field(
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
    custom_megatron_datasets_provider_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom megatron train_valid_test datasets provider function."},
    )
    custom_get_batch_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom get batch function."},
    )
    custom_loss_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom loss function."},
    )

    # remaining args such as enabling Alibi/ROPE positional embeddings,
    # wandb logging, Multi-Query Attention, etc.
    other_megatron_args: Optional[dict[str, Any]] = field(
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
        if self.recompute_activations is None:
            self.recompute_activations = str_to_bool(os.environ.get(prefix + "RECOMPUTE_ACTIVATIONS", "False")) == 1
        if self.use_distributed_optimizer is None:
            self.use_distributed_optimizer = (
                str_to_bool(os.environ.get(prefix + "USE_DISTRIBUTED_OPTIMIZER", "False")) == 1
            )
        if self.sequence_parallelism is None:
            self.sequence_parallelism = str_to_bool(os.environ.get(prefix + "SEQUENCE_PARALLELISM", "False")) == 1

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
        if self.recompute_activations:
            self.megatron_lm_default_args["recompute_granularity"] = "selective"
        if self.tensorboard_dir is not None:
            self.megatron_lm_default_args["tensorboard_dir"] = self.tensorboard_dir
            if self.set_all_logging_options:
                self.set_tensorboard_logging_options()
        if self.other_megatron_args is not None:
            self.megatron_lm_default_args.update(self.other_megatron_args)

    def set_network_size_args(self, model, batch_data=None):
        model_config_type = model.config.model_type.lower()
        for model_type in MODEL_CONFIGS_TO_MEGATRON_PARSERS.keys():
            if model_type in model_config_type:
                MODEL_CONFIGS_TO_MEGATRON_PARSERS[model_type](self, model, batch_data)
                return
        raise ValueError(
            f"Accelerate Megatron-LM integration not supports {model_config_type} model. "
            "You can add your own model config parser."
        )

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
        from megatron.training.arguments import _add_logging_args

        parser = argparse.ArgumentParser()
        parser = _add_logging_args(parser)
        logging_args = parser.parse_known_args()
        self.dataset_args = vars(logging_args[0])
        for key, value in self.dataset_args.items():
            if key.startswith("log_"):
                self.megatron_lm_default_args[key] = True
            elif key.startswith("no_log_"):
                self.megatron_lm_default_args[key.replace("no_", "")] = True


MODEL_CONFIGS_TO_MEGATRON_PARSERS = {}


def add_model_config_to_megatron_parser(model_type: str):
    def add_model_config_parser_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        MODEL_CONFIGS_TO_MEGATRON_PARSERS[model_type] = func
        return wrapper

    return add_model_config_parser_helper


@add_model_config_to_megatron_parser("megatron-bert")
def parse_bert_config(megatron_lm_plugin, model, batch_data):
    model_type_name = "bert"
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    max_position_embeddings = model.config.max_position_embeddings
    num_labels = model.config.num_labels
    orig_vocab_size = model.config.vocab_size
    pretraining_flag = False
    if "maskedlm" in model.__class__.__name__.lower():
        pretraining_flag = True
    if megatron_lm_plugin.seq_length is not None:
        if megatron_lm_plugin.encoder_seq_length is not None:
            warnings.warn("Both `seq_length` and `encoder_seq_length` are set. Using `encoder_seq_length`.")
        megatron_lm_plugin.seq_length = megatron_lm_plugin.encoder_seq_length
    elif megatron_lm_plugin.encoder_seq_length is not None:
        megatron_lm_plugin.seq_length = megatron_lm_plugin.encoder_seq_length
    elif batch_data is not None:
        megatron_lm_plugin.seq_length = batch_data["input_ids"].shape[1]
    else:
        megatron_lm_plugin.seq_length = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["seq_length"] = megatron_lm_plugin.seq_length
    megatron_lm_plugin.megatron_lm_default_args["model_type_name"] = model_type_name
    megatron_lm_plugin.megatron_lm_default_args["num_layers"] = num_layers
    megatron_lm_plugin.megatron_lm_default_args["hidden_size"] = hidden_size
    megatron_lm_plugin.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
    megatron_lm_plugin.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
    megatron_lm_plugin.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
    megatron_lm_plugin.megatron_lm_default_args["model_return_dict"] = model.config.return_dict
    megatron_lm_plugin.megatron_lm_default_args["num_labels"] = num_labels


@add_model_config_to_megatron_parser("gpt2")
def parse_gpt2_config(megatron_lm_plugin, model, batch_data):
    model_type_name = "gpt"
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    num_attention_heads = model.config.n_head
    max_position_embeddings = model.config.n_positions
    orig_vocab_size = model.config.vocab_size
    pretraining_flag = True
    if megatron_lm_plugin.seq_length is not None:
        if megatron_lm_plugin.decoder_seq_length is not None:
            warnings.warn("Both `seq_length` and `decoder_seq_length` are set. Using `decoder_seq_length`.")
        megatron_lm_plugin.seq_length = megatron_lm_plugin.decoder_seq_length
    elif megatron_lm_plugin.decoder_seq_length is not None:
        megatron_lm_plugin.seq_length = megatron_lm_plugin.decoder_seq_length
    elif batch_data is not None:
        megatron_lm_plugin.seq_length = batch_data["input_ids"].shape[1]
    else:
        megatron_lm_plugin.seq_length = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["seq_length"] = megatron_lm_plugin.seq_length
    megatron_lm_plugin.megatron_lm_default_args["return_logits"] = megatron_lm_plugin.return_logits
    megatron_lm_plugin.megatron_lm_default_args["tokenizer_type"] = "GPT2BPETokenizer"
    megatron_lm_plugin.megatron_lm_default_args["model_type_name"] = model_type_name
    megatron_lm_plugin.megatron_lm_default_args["num_layers"] = num_layers
    megatron_lm_plugin.megatron_lm_default_args["hidden_size"] = hidden_size
    megatron_lm_plugin.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
    megatron_lm_plugin.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
    megatron_lm_plugin.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
    megatron_lm_plugin.megatron_lm_default_args["model_return_dict"] = model.config.return_dict


@add_model_config_to_megatron_parser("t5")
def parse_t5_config(megatron_lm_plugin, model, batch_data):
    model_type_name = "t5"
    num_layers = model.config.num_layers
    hidden_size = model.config.d_model
    num_attention_heads = model.config.num_heads
    max_position_embeddings = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
    orig_vocab_size = model.config.vocab_size
    pretraining_flag = True
    if megatron_lm_plugin.encoder_seq_length is None:
        if batch_data is not None:
            megatron_lm_plugin.encoder_seq_length = batch_data["input_ids"].shape[1]
        else:
            megatron_lm_plugin.encoder_seq_length = max_position_embeddings
    if megatron_lm_plugin.decoder_seq_length is None:
        if batch_data is not None:
            megatron_lm_plugin.decoder_seq_length = batch_data["labels"].shape[1]
        else:
            megatron_lm_plugin.decoder_seq_length = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["encoder_seq_length"] = megatron_lm_plugin.encoder_seq_length
    megatron_lm_plugin.megatron_lm_default_args["decoder_seq_length"] = megatron_lm_plugin.decoder_seq_length
    megatron_lm_plugin.megatron_lm_default_args["model_type_name"] = model_type_name
    megatron_lm_plugin.megatron_lm_default_args["num_layers"] = num_layers
    megatron_lm_plugin.megatron_lm_default_args["hidden_size"] = hidden_size
    megatron_lm_plugin.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
    megatron_lm_plugin.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
    megatron_lm_plugin.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
    megatron_lm_plugin.megatron_lm_default_args["model_return_dict"] = model.config.return_dict


@add_model_config_to_megatron_parser("llama")
def parse_llama_config(megatron_lm_plugin, model, batch_data):
    model_type_name = "gpt"
    num_layers = model.config.num_hidden_layers
    pretraining_flag = True
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    orig_vocab_size = model.config.vocab_size

    max_position_embeddings = model.config.max_position_embeddings
    seq_length = getattr(model.config, "max_sequence_length", None)
    if megatron_lm_plugin.seq_length is None:
        if seq_length is not None:
            megatron_lm_plugin.seq_length = seq_length
        elif megatron_lm_plugin.decoder_seq_length is not None:
            megatron_lm_plugin.seq_length = megatron_lm_plugin.decoder_seq_length
        elif batch_data is not None:
            megatron_lm_plugin.seq_length = batch_data["input_ids"].shape[1]
        else:
            megatron_lm_plugin.seq_length = max_position_embeddings

    megatron_lm_plugin.megatron_lm_default_args["return_logits"] = megatron_lm_plugin.return_logits
    megatron_lm_plugin.megatron_lm_default_args["tokenizer_type"] = "Llama2Tokenizer"
    megatron_lm_plugin.megatron_lm_default_args["model_type_name"] = model_type_name
    megatron_lm_plugin.megatron_lm_default_args["num_layers"] = num_layers
    megatron_lm_plugin.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
    megatron_lm_plugin.megatron_lm_default_args["hidden_size"] = hidden_size
    megatron_lm_plugin.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
    megatron_lm_plugin.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
    megatron_lm_plugin.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
    megatron_lm_plugin.megatron_lm_default_args["seq_length"] = megatron_lm_plugin.seq_length
    megatron_lm_plugin.megatron_lm_default_args["model_return_dict"] = model.config.return_dict


@dataclass
class BnbQuantizationConfig:
    """
    A plugin to enable BitsAndBytes 4bit and 8bit quantization

    Args:
        load_in_8bit (`bool`, defaults to `False`):
            Enable 8bit quantization.
        llm_int8_threshold (`float`, defaults to `6.0`):
            Value of the outliner threshold. Only relevant when `load_in_8bit=True`.
        load_in_4_bit (`bool`, defaults to `False`):
            Enable 4bit quantization.
        bnb_4bit_quant_type (`str`, defaults to `fp4`):
            Set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','np4'}.
        bnb_4bit_use_double_quant (`bool`, defaults to `False`):
            Enable nested quantization where the quantization constants from the first quantization are quantized
            again.
        bnb_4bit_compute_dtype (`bool`, defaults to `fp16`):
            This sets the computational type which might be different than the input time. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}.
        torch_dtype (`torch.dtype`, defaults to `None`):
            This sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value
            to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model.
        skip_modules (`List[str]`, defaults to `None`):
            An explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`.
        keep_in_fp32_modules (`List`, defaults to `None`):
            An explicit list of the modules that we don't quantize. We keep them in `torch.float32`.
    """

    load_in_8bit: bool = field(default=False, metadata={"help": "enable 8bit quantization."})

    llm_int8_threshold: float = field(
        default=6.0,
        metadata={"help": "value of the outliner threshold. only relevant when load_in_8bit=True"},
    )

    load_in_4bit: bool = field(default=False, metadata={"help": "enable 4bit quantization."})

    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','nf4'}."
        },
    )

    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "enable nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )

    bnb_4bit_compute_dtype: str = field(
        default="fp16",
        metadata={
            "help": "This sets the computational type which might be different than the input time. For example, inputs might be "
            "fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}."
        },
    )

    torch_dtype: torch.dtype = field(
        default=None,
        metadata={
            "help": "this sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value"
            "to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model "
        },
    )

    skip_modules: list[str] = field(
        default=None,
        metadata={
            "help": "an explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`."
        },
    )

    keep_in_fp32_modules: list[str] = field(
        default=None,
        metadata={"help": "an explicit list of the modules that we don't quantize. We keep them in `torch.float32`."},
    )

    def __post_init__(self):
        """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_8bit, bool):
            raise ValueError("load_in_8bit must be a boolean")

        if not isinstance(self.load_in_4bit, bool):
            raise ValueError("load_in_4bit must be a boolean")

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit can't be both True")

        if not self.load_in_4bit and not self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit can't be both False")

        if not isinstance(self.llm_int8_threshold, (int, float)):
            raise ValueError("llm_int8_threshold must be a float or an int")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError("bnb_4bit_quant_type must be a string")
        elif self.bnb_4bit_quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"bnb_4bit_quant_type must be in ['fp4','nf4'] but found {self.bnb_4bit_quant_type}")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError("bnb_4bit_use_double_quant must be a boolean")

        if isinstance(self.bnb_4bit_compute_dtype, str):
            if self.bnb_4bit_compute_dtype == "fp32":
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.bnb_4bit_compute_dtype == "fp16":
                self.bnb_4bit_compute_dtype = torch.float16
            elif self.bnb_4bit_compute_dtype == "bf16":
                self.bnb_4bit_compute_dtype = torch.bfloat16
            else:
                raise ValueError(
                    f"bnb_4bit_compute_dtype must be in ['fp32','fp16','bf16'] but found {self.bnb_4bit_compute_dtype}"
                )
        elif not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if self.skip_modules is not None and not isinstance(self.skip_modules, list):
            raise ValueError("skip_modules must be a list of strings")

        if self.keep_in_fp32_modules is not None and not isinstance(self.keep_in_fp32_modules, list):
            raise ValueError("keep_in_fp_32_modules must be a list of strings")

        if self.load_in_4bit:
            self.target_dtype = CustomDtype.INT4

        if self.load_in_8bit:
            self.target_dtype = torch.int8

        if self.load_in_4bit and self.llm_int8_threshold != 6.0:
            warnings.warn("llm_int8_threshold can only be used for model loaded in 8bit")

        if isinstance(self.torch_dtype, str):
            if self.torch_dtype == "fp32":
                self.torch_dtype = torch.float32
            elif self.torch_dtype == "fp16":
                self.torch_dtype = torch.float16
            elif self.torch_dtype == "bf16":
                self.torch_dtype = torch.bfloat16
            else:
                raise ValueError(f"torch_dtype must be in ['fp32','fp16','bf16'] but found {self.torch_dtype}")
        if self.load_in_8bit and self.torch_dtype is None:
            self.torch_dtype = torch.float16

        if self.load_in_4bit and self.torch_dtype is None:
            self.torch_dtype = self.bnb_4bit_compute_dtype

        if not isinstance(self.torch_dtype, torch.dtype):
            raise ValueError("torch_dtype must be a torch.dtype")


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
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
