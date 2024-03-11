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

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union

import torch
import torch.utils.hooks as hooks

from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
    MODEL_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    AutocastKwargs,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    DynamoBackend,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProjectConfiguration,
    RNGType,
    TorchDynamoPlugin,
    check_os_kernel,
    clean_state_dict_for_safetensors,
    compare_versions,
    convert_model,
    convert_outputs_to_fp32,
    extract_model_from_parallel,
    gather,
    gather_object,
    get_mixed_precision_context_manager,
    get_pretty_name,
    has_transformer_engine_layers,
    is_bf16_available,
    is_deepspeed_available,
    is_fp8_available,
    is_ipex_available,
    is_megatron_lm_available,
    is_msamp_available,
    is_npu_available,
    is_torch_version,
    is_torch_xla_available,
    is_xpu_available,
    load_fsdp_model,
    load_fsdp_optimizer,
    pad_across_processes,
    parse_choice_from_env,
    recursively_apply,
    reduce,
    release_memory,
    save,
    save_fsdp_model,
    save_fsdp_optimizer,
    shard_checkpoint,
    wait_for_everyone,
)
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module


if is_deepspeed_available():
    from .utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

if is_fp8_available():
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast


if is_megatron_lm_available():
    from .utils import (
        MegatronEngine,
        MegatronLMDummyDataLoader,
        MegatronLMDummyScheduler,
        MegatronLMOptimizerWrapper,
        MegatronLMSchedulerWrapper,
        megatron_lm_initialize,
        megatron_lm_prepare_data_loader,
        megatron_lm_prepare_model,
        megatron_lm_prepare_optimizer,
        megatron_lm_prepare_scheduler,
    )

from torch.distributed.algorithms.join import Join


if is_torch_xla_available():
    import torch_xla.amp as xamp
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


if is_npu_available(check_device=False):
    import torch_npu  # noqa: F401


try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

logger = get_logger(__name__)

# Sentinel values for defaults
_split_batches = object()
_dispatch_batches = object()
_even_batches = object()
_use_seedable_sampler = object()


class Accelerator:
    """
    Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

    Args:
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
            etc...).
        mixed_precision (`str`, *optional*):
            Whether or not to use mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'. Will default to the
            value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the
            accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp8'
            requires the installation of transformers-engine.
        gradient_accumulation_steps (`int`, *optional*, default to 1):
            The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
            `Accelerator.accumulate`. If not passed, will default to the value in the environment variable
            `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
            the execution on one process only.
        dataloader_config (`DataLoaderConfiguration`, *optional*):
            A configuration for how the dataloaders should be handled in distributed scenarios.
        deepspeed_plugin ([`~utils.DeepSpeedPlugin`], *optional*):
            Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        fsdp_plugin ([`~utils.FullyShardedDataParallelPlugin`], *optional*):
            Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
            using *accelerate config*
        megatron_lm_plugin ([`~utils.MegatronLMPlugin`], *optional*):
            Tweak your MegatronLM related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration in your prepared
            dataloaders. Should be one or several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

            Will default to `["torch"]` for PyTorch versions <=1.5.1 and `["generator"]` for PyTorch versions >= 1.6.
        log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
            A list of loggers to be setup for experiment tracking. Should be one or several of:

            - `"all"`
            - `"tensorboard"`
            - `"wandb"`
            - `"comet_ml"`
            If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
            also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        project_config ([`~utils.ProjectConfiguration`], *optional*):
            A configuration for how saving the state can be handled.
        project_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing data such as logs of locally-compatible loggers and potentially saved
            checkpoints.
        step_scheduler_with_optimizer (`bool`, *optional`, defaults to `True`):
            Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
            done under certain circumstances (at the end of each epoch, for instance).
        kwargs_handlers (list of [`~utils.KwargsHandler`], *optional*)
            A list of [`~utils.KwargsHandler`] to customize how the objects related to distributed training or mixed
            precision are created. See [kwargs](kwargs) for more information.
        dynamo_backend (`str` or [`~utils.DynamoBackend`], *optional*, defaults to `"no"`):
            Set to one of the possible dynamo backends to optimize your training with torch dynamo.
        gradient_accumulation_plugin ([`~utils.GradientAccumulationPlugin`], *optional*):
            A configuration for how gradient accumulation should be handled, if more tweaking than just the
            `gradient_accumulation_steps` is needed.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~utils.DistributedType`]) -- The distributed training configuration.
        - **local_process_index** (`int`) -- The process index on the current machine.
        - **mixed_precision** (`str`) -- The configured mixed precision mode.
        - **num_processes** (`int`) -- The total number of processes used for training.
        - **optimizer_step_was_skipped** (`bool`) -- Whether or not the optimizer update was skipped (because of
          gradient overflow in mixed precision), in which
        case the learning rate should not be changed.
        - **process_index** (`int`) -- The overall index of the current process among all processes.
        - **state** ([`~state.AcceleratorState`]) -- The distributed setup state.
        - **sync_gradients** (`bool`) -- Whether the gradients are currently being synced across all processes.
        - **use_distributed** (`bool`) -- Whether the current configuration is for distributed training.
    """

    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = _split_batches,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        dataloader_config: DataLoaderConfiguration | None = None,
        deepspeed_plugin: DeepSpeedPlugin | None = None,
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
        megatron_lm_plugin: MegatronLMPlugin | None = None,
        rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        dispatch_batches: bool | None = _dispatch_batches,
        even_batches: bool = _even_batches,
        use_seedable_sampler: bool = _use_seedable_sampler,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
        dynamo_backend: DynamoBackend | str | None = None,
    ):
        self.trackers = []
        if project_config is not None:
            self.project_configuration = project_config
        else:
            self.project_configuration = ProjectConfiguration(project_dir=project_dir)
        if project_dir is not None and self.project_dir is None:
            self.project_configuration.set_directories(project_dir)
        if mixed_precision is not None:
            mixed_precision = str(mixed_precision)
            if mixed_precision not in PrecisionType:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
                )

        dynamo_plugin = TorchDynamoPlugin() if dynamo_backend is None else TorchDynamoPlugin(backend=dynamo_backend)

        if deepspeed_plugin is None:  # init from env variables
            deepspeed_plugin = (
                DeepSpeedPlugin() if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" else None
            )
        else:
            assert isinstance(
                deepspeed_plugin, DeepSpeedPlugin
            ), "`deepspeed_plugin` must be an `accelerate.utils.DeepSpeedPlugin` object."
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"  # use DeepSpeed if plugin is provided
        if deepspeed_plugin:
            if not is_deepspeed_available():
                raise ImportError("DeepSpeed is not installed => run `pip install deepspeed` or build it from source.")
            if compare_versions("deepspeed", "<", "0.9.3"):
                raise ImportError("DeepSpeed version must be >= 0.9.3. Please update DeepSpeed.")

            mixed_precision = (
                os.environ.get("ACCELERATE_MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
            )
            deepspeed_plugin.set_mixed_precision(mixed_precision)
            deepspeed_plugin.set_deepspeed_weakref()

        if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" or isinstance(
            fsdp_plugin, FullyShardedDataParallelPlugin
        ):
            if is_torch_version("<", FSDP_PYTORCH_VERSION):
                raise ValueError(f"FSDP requires PyTorch >= {FSDP_PYTORCH_VERSION}")

        if fsdp_plugin is None:  # init from env variables
            fsdp_plugin = (
                FullyShardedDataParallelPlugin() if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" else None
            )
        else:
            if not isinstance(fsdp_plugin, FullyShardedDataParallelPlugin):
                raise TypeError("`fsdp_plugin` must be a FullyShardedDataParallelPlugin object.")
            os.environ["ACCELERATE_USE_FSDP"] = "true"  # use FSDP if plugin is provided

        if megatron_lm_plugin is None:  # init from env variables
            megatron_lm_plugin = (
                MegatronLMPlugin() if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true" else None
            )
        else:
            if not isinstance(megatron_lm_plugin, MegatronLMPlugin):
                raise TypeError("`megatron_lm_plugin` must be a MegatronLMPlugin object.")
            os.environ["ACCELERATE_USE_MEGATRON_LM"] = "true"  # use MegatronLM if plugin is provided

        if megatron_lm_plugin:
            if not is_megatron_lm_available():
                raise ImportError("Megatron is not installed. please build it from source.")

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        self.fp8_recipe_handler = None
        self.autocast_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(
                    handler, KwargsHandler
                ), f"Unsupported kwargs handler passed: {handler}, must be one that inherits `accelerate.utils.KwargsHandler`."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError("You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`.")
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError("You can only pass one `GradScalerKwargs` in `kwargs_handler`.")
                    else:
                        self.scaler_handler = handler
                elif isinstance(handler, InitProcessGroupKwargs):
                    if self.init_handler is not None:
                        raise ValueError("You can only pass one `InitProcessGroupKwargs` in `kwargs_handler`.")
                    else:
                        self.init_handler = handler
                elif isinstance(handler, FP8RecipeKwargs):
                    if self.fp8_recipe_handler is not None:
                        raise ValueError("You can only pass one `FP8RecipeKwargs` in `kwargs_handler`.")
                    else:
                        self.fp8_recipe_handler = handler
                elif isinstance(handler, AutocastKwargs):
                    if self.autocast_handler is not None:
                        raise ValueError("You can only pass one `AutocastKwargs` in `kwargs_handler`.")
                    else:
                        self.autocast_handler = handler

        kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
        self.state = AcceleratorState(
            mixed_precision=mixed_precision,
            cpu=cpu,
            dynamo_plugin=dynamo_plugin,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            megatron_lm_plugin=megatron_lm_plugin,
            _from_accelerator=True,
            **kwargs,
        )

        if self.fp8_recipe_handler is None and self.state.mixed_precision == "fp8":
            self.fp8_recipe_handler = FP8RecipeKwargs(backend="MSAMP" if is_msamp_available() else "TE")

        trackers = filter_trackers(log_with, self.logging_dir)
        if len(trackers) < 1 and log_with is not None:
            warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
        self.log_with = trackers

        if (
            (mixed_precision != "bf16")
            and getattr(self.state, "downcast_bfloat", False)
            and (self.state.distributedType != DistributedType.XLA)
        ):
            raise ValueError("Can only use `downcast_bf16` when using `mixed_precision='bf16'` and on a TPU")

        if gradient_accumulation_plugin is not None:
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "You can only pass one of `gradient_accumulation_steps` and `gradient_accumulation_plugin`. Please only pass in the created `GradientAccumulationPlugin` object."
                )
        else:
            gradient_accumulation_steps = int(
                parse_choice_from_env("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps)
            )
            gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps)
        self.gradient_state = GradientState(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.device_placement = device_placement
        if dataloader_config is None:
            dataloader_config = DataLoaderConfiguration()
        self.dataloader_config = dataloader_config
        # Deal with deprecated args
        # TODO: Remove in v1.0.0
        deprecated_dl_args = {}
        if dispatch_batches is not _dispatch_batches:
            deprecated_dl_args["dispatch_batches"] = dispatch_batches
            self.dataloader_config.dispatch_batches = dispatch_batches
        if split_batches is not _split_batches:
            deprecated_dl_args["split_batches"] = split_batches
            self.dataloader_config.split_batches = split_batches
        if even_batches is not _even_batches:
            deprecated_dl_args["even_batches"] = even_batches
            self.dataloader_config.even_batches = even_batches
        if use_seedable_sampler is not _use_seedable_sampler:
            deprecated_dl_args["use_seedable_sampler"] = use_seedable_sampler
            self.dataloader_config.use_seedable_sampler = use_seedable_sampler
        if len(deprecated_dl_args) > 0:
            values = ", ".join([f"{k}={v}" for k, v in deprecated_dl_args.items()])
            warnings.warn(
                f"Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: {deprecated_dl_args.keys()}. "
                "Please pass an `accelerate.DataLoaderConfiguration` instead: \n"
                f"dataloader_config = DataLoaderConfiguration({values})",
                FutureWarning,
            )
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if (
            self.state.mixed_precision == "fp16"
            and self.device.type != "cpu"
            and self.distributed_type not in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM)
        ):
            self.native_amp = True
            if self.device.type not in ("xpu", "cuda", "mps", "npu", "xla") or is_torch_xla_available(
                check_is_tpu=True
            ):
                raise ValueError(f"fp16 mixed precision requires a GPU (not {self.device.type!r}).")
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            if self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler(**kwargs)
            elif is_torch_xla_available(check_is_gpu=True):
                self.scaler = xamp.GradScaler(**kwargs)
            elif is_npu_available():
                self.scaler = torch.npu.amp.GradScaler(**kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)

        elif self.state.mixed_precision == "bf16" and self.distributed_type not in (
            DistributedType.DEEPSPEED,
            DistributedType.MEGATRON_LM,
        ):
            if self.device.type in ["cpu", "xpu"]:
                self.native_amp = True
            else:
                self.native_amp = is_bf16_available(True)
            if mixed_precision == "bf16" and not self.native_amp and not is_torch_xla_available():
                raise ValueError("bf16 mixed precision requires PyTorch >= 1.10 and a supported device.")

        # Start of internal step tracking
        self.step = 0

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._dataloaders = []
        self._custom_objects = []

        # Hooks
        self._load_model_state_pre_hook = OrderedDict()
        self._save_model_state_pre_hook = OrderedDict()

        # RNG Types
        self.rng_types = rng_types
        if self.rng_types is None:
            self.rng_types = ["generator"]

        # Set a flag tensor for early stopping and other breakpoints
        self.flag_tensor = None

        check_os_kernel()

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.state.use_distributed

    @property
    def distributed_type(self):
        return self.state.distributed_type

    @property
    def num_processes(self):
        return self.state.num_processes

    @property
    def process_index(self):
        return self.state.process_index

    @property
    def local_process_index(self):
        return self.state.local_process_index

    @property
    def device(self):
        return self.state.device

    @property
    def split_batches(self):
        return self.dataloader_config.split_batches

    @property
    def dispatch_batches(self):
        return self.dataloader_config.dispatch_batches

    @property
    def even_batches(self):
        return self.dataloader_config.even_batches

    @even_batches.setter
    def even_batches(self, value: bool):
        self.dataloader_config.even_batches = value

    @property
    def use_seedable_sampler(self):
        return self.dataloader_config.use_seedable_sampler

    @property
    def project_dir(self):
        return self.project_configuration.project_dir

    @property
    def logging_dir(self):
        return self.project_configuration.logging_dir

    @property
    def save_iteration(self):
        return self.project_configuration.iteration

    @property
    def is_main_process(self):
        """True for one process only."""
        return self.state.is_main_process

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.state.is_local_main_process

    @property
    def use_fp16(self):
        warnings.warn(
            "The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use "
            "`Accelerator.mixed_precision == 'fp16'` instead.",
            FutureWarning,
        )
        return self.mixed_precision != "no"

    @property
    def is_last_process(self):
        return self.process_index == self.num_processes - 1

    @property
    def mixed_precision(self):
        return self.state.mixed_precision

    @contextmanager
    def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool = False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `Accelerator.gather()` on the outputs
                or passing in less inputs than there are processes. If so, just remember to drop the padded elements
                afterwards.

        Example:

        ```python
        # Assume there are two processes
        from accelerate import Accelerator

        accelerator = Accelerator()
        with accelerator.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with accelerator.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        with PartialState().split_between_processes(inputs, apply_padding=apply_padding) as inputs:
            yield inputs

    def on_main_process(self, function: Callable[..., Any] = None):
        """
        A decorator that will run the decorated function on the main process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()


        >>> @accelerator.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_main_process(function)(*args, **kwargs)

        return _inner

    def on_local_main_process(self, function: Callable[..., Any] = None):
        """
        A decorator that will run the decorated function on the local main process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_local_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_local_main_process(function)(*args, **kwargs)

        return _inner

    def on_last_process(self, function: Callable[..., Any]):
        """
        A decorator that will run the decorated function on the last process only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_last_process
        def print_something():
            print(f"Printed on process {accelerator.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_last_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_last_process(function)(*args, **kwargs)

        return _inner

    def on_process(self, function: Callable[..., Any] = None, process_index: int = None):
        """
        A decorator that will run the decorated function on a given process index only. Can also be called using the
        `PartialState` class.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        # Initial construction of the decorator.
        if (self is not None) and (process_index is not None) and (function is None):
            return partial(self.on_process, process_index=process_index)
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_process(function, process_index)(*args, **kwargs)

        return _inner

    def on_local_process(self, function: Callable[..., Any] = None, local_process_index: int = None):
        """
        A decorator that will run the decorated function on a given local process index only. Can also be called using
        the `PartialState` class.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        # Initial construction of the decorator.
        if (self is not None) and (local_process_index is not None) and (function is None):
            return partial(self.on_local_process, local_process_index=local_process_index)
        # For times when the `Accelerator` object itself utilizes this decorator.
        if function is None:
            if "Accelerator." in self.__qualname__:
                function = self
            else:
                raise ValueError(
                    "The `on_main_process` decorator must be called with a function on an instantiated `Accelerator` object."
                )

        def _inner(*args, **kwargs):
            return PartialState().on_local_process(function, local_process_index)(*args, **kwargs)

        return _inner

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        with self.state.main_process_first():
            yield

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.local_process_index}")
        ```
        """
        with self.state.local_main_process_first():
            yield

    @contextmanager
    def no_sync(self, model):
        """
        A context manager to disable gradient synchronizations across DDP processes by calling
        `torch.nn.parallel.DistributedDataParallel.no_sync`.

        If `model` is not in DDP, this context manager does nothing

        Args:
            model (`torch.nn.Module`):
                PyTorch Module that was prepared with `Accelerator.prepare`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
        >>> input_a = next(iter(dataloader))
        >>> input_b = next(iter(dataloader))

        >>> with accelerator.no_sync():
        ...     outputs = model(input_a)
        ...     loss = loss_func(outputs)
        ...     accelerator.backward(loss)
        ...     # No synchronization across processes, only accumulate gradients
        >>> outputs = model(input_b)
        >>> accelerator.backward(loss)
        >>> # Synchronization across all processes
        >>> optimizer.step()
        >>> optimizer.zero_grad()
        ```
        """
        context = contextlib.nullcontext
        if self.use_distributed:
            context = getattr(model, "no_sync", context)

        with context():
            yield

    @staticmethod
    @contextmanager
    def trigger_sync_in_backward(model):
        """Trigger the sync of the gradients in the next backward pass of the model after multiple forward passes under
        `Accelerator.no_sync` (only applicable in multi-GPU scenarios).

                If the script is not launched in distributed mode, this context manager does nothing.

                Args:
                    model (`torch.nn.Module`):
                        The model for which to trigger the gradient synchronization.

                Example:

                ```python
                >>> from accelerate import Accelerator

                >>> accelerator = Accelerator()
                >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

                >>> with accelerator.no_sync():
                ...     loss_a = loss_func(model(input_a))  # first forward pass
                ...     loss_b = loss_func(model(input_b))  # second forward pass
                >>> accelerator.backward(loss_a)  # No synchronization across processes, only accumulate gradients
                >>> with accelerator.trigger_sync_in_backward(model):
                ...     accelerator.backward(loss_b)  # Synchronization across all processes
                >>> optimizer.step()
                >>> optimizer.zero_grad()
                ```
        """
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            yield
            return

        old_require_backward_grad_sync = model.require_backward_grad_sync
        old_require_forward_param_sync = model.require_forward_param_sync

        # EXPERIMENTAL: This will force grad sync during `backward()`, but it is unknown if it breaks other DDP features.
        # https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/torch/nn/parallel/distributed.py#L1453-L1466
        model.require_backward_grad_sync = True
        model.require_forward_param_sync = True
        # https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/torch/csrc/distributed/c10d/reducer.cpp#L1371-L1402
        model.reducer.prepare_for_backward([])
        try:
            yield
        finally:
            model.require_backward_grad_sync = old_require_backward_grad_sync
            model.require_forward_param_sync = old_require_forward_param_sync

    def _do_sync(self, force: bool = False):
        "Sets the right `sync_gradients` context and either resets or increases `self.step`"
        if self.gradient_state.sync_with_dataloader and self.gradient_state.end_of_dataloader:
            self.step = 0
            self.gradient_state._set_sync_gradients(True)
        else:
            self.step += 1
            self.gradient_state._set_sync_gradients(force or ((self.step % self.gradient_state.num_steps) == 0))

    @property
    def sync_gradients(self):
        return self.gradient_state.sync_gradients

    @sync_gradients.setter
    def sync_gradients(self, sync_gradients):
        self.gradient_state.sync_gradients = sync_gradients

    @property
    def gradient_accumulation_steps(self):
        return self.gradient_state.num_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, gradient_accumulation_steps):
        self.gradient_state.plugin_kwargs.update({"num_steps": gradient_accumulation_steps})

    @contextmanager
    def accumulate(self, *models):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `torch.nn.Module`):
                PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
                skip gradient syncing during backward pass in distributed training

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=1)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, output in dataloader:
        ...     with accelerator.accumulate(model):
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
        # sync_each_batch=True will guarantee below that self.sync_gradients=True, therefore
        # resulting in the nullcontext always being selected.
        self._do_sync(force=self.gradient_state.plugin_kwargs.get("sync_each_batch", False))
        with contextlib.ExitStack() as cm_stack:
            for m in models:
                cm_stack.enter_context(contextlib.nullcontext() if self.sync_gradients else self.no_sync(m))
            yield

    @contextmanager
    def join_uneven_inputs(self, joinables, even_batches=None):
        """
        A context manager that facilitates distributed training or evaluation on uneven inputs, which acts as a wrapper
        around `torch.distributed.algorithms.join`. This is useful when the total batch size does not evenly divide the
        length of the dataset.

        Args:
            joinables (`list[torch.distributed.algorithms.Joinable]`):
                A list of models or optimizers that subclass `torch.distributed.algorithms.Joinable`. Most commonly, a
                PyTorch Module that was prepared with `Accelerator.prepare` for DistributedDataParallel training.
            even_batches (`bool`, *optional*)
                If set, this will override the value of `even_batches` set in the `Accelerator`. If it is not provided,
                the default `Accelerator` value wil be used.

        <Tip warning={true}>

        `join_uneven_inputs` is only supported for Distributed Data Parallel training on multiple GPUs. For any other
        configuration, this method will have no effect.

        </Tip>

        <Tip warning={true}>

        Overidding `even_batches` will not affect iterable-style data loaders.

        </Tip>

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(even_batches=True)
        >>> ddp_model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        >>> with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
        ...     for input, output in dataloader:
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         optimizer.zero_grad()
        ```
        """
        if self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_NPU, DistributedType.MULTI_XPU):
            dl_even_batches_values = []

            if even_batches is not None:
                iterable_dl_seen = False
                # override value in batch sampler for map-style datasets
                for dl_idx, dl in enumerate(self._dataloaders):
                    if isinstance(dl, DataLoaderDispatcher):
                        iterable_dl_seen = True
                        continue
                    dl_even_batches_values.append((dl_idx, dl.batch_sampler.even_batches))
                    dl.batch_sampler.even_batches = even_batches

                if iterable_dl_seen:
                    warnings.warn(
                        "Overridding even_batches is only supported for map-style datasets, yet some dataloaders given were iterable"
                    )
            else:
                even_batches = self.even_batches

            enable_join = False if even_batches else True
            try:
                with Join(joinables, enable=enable_join, throw_on_early_termination=False):
                    yield
            finally:
                # reset any batch samplers that have been modified
                for dl_idx, even_batches_value in dl_even_batches_values:
                    self._dataloaders[dl_idx].batch_sampler.even_batches = even_batches_value
        else:
            # Even when disabled, Join expects models to subclass Joinable, so skip entirely for single process runs
            if self.distributed_type != DistributedType.NO:
                warnings.warn(
                    "Joining uneven inputs is only supported for multi-GPU training, as a result `join_uneven_inputs` will have no effect."
                )

            with contextlib.nullcontext(joinables):
                yield

    def print(self, *args, **kwargs):
        """
        Drop in replacement of `print()` to only print once per server.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> accelerator.print("Hello world!")
        ```
        """
        self.state.print(*args, **kwargs)

    def _prepare_one(self, obj, first_pass=False, device_placement=None):
        # First pass of preparation: DataLoader, model, optimizer
        if first_pass:
            if isinstance(obj, torch.utils.data.DataLoader):
                return self.prepare_data_loader(obj, device_placement=device_placement)
            elif isinstance(obj, torch.nn.Module):
                return self.prepare_model(obj, device_placement=device_placement)
            elif isinstance(obj, torch.optim.Optimizer):
                optimizer = self.prepare_optimizer(obj, device_placement=device_placement)
                return optimizer
        # Second pass of preparation: LR scheduler (which need the full list of optimizers)
        elif isinstance(obj, LRScheduler):
            scheduler = self.prepare_scheduler(obj)
            return scheduler
        # Return the unprocessed object if previous criteria was not met
        return obj

    def prepare(self, *args, device_placement=None):
        """
        Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
        order.

        Args:
            *args (list of objects):
                Any of the following type of objects:

                - `torch.utils.data.DataLoader`: PyTorch Dataloader
                - `torch.nn.Module`: PyTorch Module
                - `torch.optim.Optimizer`: PyTorch Optimizer
                - `torch.optim.lr_scheduler.LRScheduler`: PyTorch LR Scheduler

            device_placement (`list[bool]`, *optional*):
                Used to customize whether automatic device placement should be performed for each object passed. Needs
                to be a list of the same length as `args`. Not compatible with DeepSpeed or FSDP.

        <Tip>

          You don't need to prepare a model if you only use it for inference without any kind of mixed precision

        </Tip>

        Examples:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model, optimizer, data_loader and scheduler are defined
        >>> model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
        ```

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model, optimizer, data_loader and scheduler are defined
        >>> device_placement = [True, True, False, False]
        >>> # Will place the first to items passed in automatically to the right device but not the last two.
        >>> model, optimizer, data_loader, scheduler = accelerator.prepare(
        ...     model, optimizer, data_loader, scheduler, device_placement=device_placement
        ... )
        ```
        """
        if device_placement is None:
            device_placement = [None for _ in args]
        elif self.distributed_type in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM):
            raise ValueError("You can't customize device placements with DeepSpeed or Megatron-LM.")
        elif len(device_placement) != len(args):
            raise ValueError(
                f"`device_placement` should be a list with {len(args)} elements (the number of objects passed)."
            )

        for obj in args:
            # TODO: Look at enabling native TP training directly with a proper config
            if (
                isinstance(obj, torch.nn.Module)
                and self.verify_device_map(obj)
                and self.distributed_type != DistributedType.NO
                and os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true"
            ):
                raise ValueError(
                    "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                    " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
                )

        if self.distributed_type == DistributedType.DEEPSPEED:
            model_count = 0
            for obj in args:
                if isinstance(obj, torch.nn.Module):
                    model_count += 1
            if model_count > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                )

        # On TPUs, putting the model on the XLA device will create new parameters, so the corresponding optimizer will
        # have parameters disconnected from the model (so no training :-( ).
        # If the model and optimizer have parameters on different devices we raise an error.
        if self.distributed_type == DistributedType.XLA:
            model_device, optimizer_device = self._get_devices()
            if model_device is not None and optimizer_device is not None and model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you "
                    "created an optimizer around your model **before** putting on the device. Make sure the line "
                    "model.to(device) is before the optimizer creation in your script or remove it entirely and use "
                    "the flag default value for `device_placement` in your `Accelerator` to let it handle that "
                    "part for you."
                )

        # If we're dealing with device placement, this deals with that by...
        tpu_should_fix_optimizer = self.device_placement and self.distributed_type == DistributedType.XLA
        if tpu_should_fix_optimizer or (self.mixed_precision == "fp8" and self.fp8_recipe_handler.backend == "TE"):
            # 1. grabbing old model parameters
            old_named_params = self._get_named_parameters(*args)

        if self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
            if self.device.type == "cpu" and self.state.use_ipex:
                args = self._prepare_ipex(*args)
            elif self.device.type == "xpu" and is_xpu_available():
                args = self._prepare_ipex(*args)
        if self.distributed_type == DistributedType.DEEPSPEED:
            result = self._prepare_deepspeed(*args)
        elif self.distributed_type == DistributedType.MEGATRON_LM:
            result = self._prepare_megatron_lm(*args)
        else:
            if self.mixed_precision == "fp8" and self.fp8_recipe_handler.backend == "MSAMP":
                args = self._prepare_msamp(*args)
                # MS-AMP will handle the device placement
                device_placement = [False for _ in args]
            result = tuple(
                self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
            )
            result = tuple(self._prepare_one(obj, device_placement=d) for obj, d in zip(result, device_placement))

        if tpu_should_fix_optimizer or (self.mixed_precision == "fp8" and self.fp8_recipe_handler.backend == "TE"):
            # 2. grabbing new model parameters
            new_named_params = self._get_named_parameters(*result)
            # 3. building a map from the first to the second
            mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
            # 4. using that map to update the parameters of the optimizer
            for obj in result:
                if isinstance(obj, torch.optim.Optimizer):
                    obj._switch_parameters(mapping)

        for item in result:
            if any(
                item in container
                for container in (self._dataloaders, self._models, self._optimizers, self._schedulers)
            ):
                item._is_accelerate_prepared = True

        return result if len(result) > 1 else result[0]

    def prepare_model(self, model: torch.nn.Module, device_placement: bool = None, evaluation_mode: bool = False):
        """
        Prepares a PyTorch model for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model to prepare. You don't need to prepare a model if it is used only for inference without
                any kind of mixed precision
            device_placement (`bool`, *optional*):
                Whether or not to place the model on the proper device. Will default to `self.device_placement`.
            evaluation_mode (`bool`, *optional*, defaults to `False`):
                Whether or not to set the model for evaluation only, by just applying mixed precision and
                `torch.compile` (if configured in the `Accelerator` object).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model is defined
        >>> model = accelerator.prepare_model(model)
        ```
        """
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
        self._models.append(model)

        # TODO: Look at enabling native TP training directly with a proper config
        if (
            self.verify_device_map(model)
            and self.distributed_type != DistributedType.NO
            and os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true"
        ):
            raise ValueError(
                "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
            )

        if self.native_amp:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
            autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
            new_forward = autocast_context(model_forward_func)
            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
            else:
                model.forward = convert_outputs_to_fp32(new_forward)
        elif self.mixed_precision == "fp8" and self.fp8_recipe_handler.backend == "TE":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            model.forward = fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)(model.forward)

        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(
            model, "hf_device_map", False
        ):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1 and self.distributed_type != DistributedType.NO:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices in any distributed mode."
                    " In order to use 8-bit models that have been loaded across multiple GPUs the solution is to use Naive Pipeline Parallelism."
                    " Therefore you should not specify that you are under any distributed regime in your accelerate config."
                )
            current_device = list(model_devices)[0]
            current_device_index = current_device.index if isinstance(current_device, torch.device) else current_device

            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not self.verify_device_map(model):
            model = model.to(self.device)
        if not evaluation_mode:
            if self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
                DistributedType.MULTI_XPU,
            ):
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    # TODO: Look at enabling native TP training directly with a proper config
                    if os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true":
                        device_ids, output_device = [self.local_process_index], self.local_process_index
                    else:
                        device_ids, output_device = None, None

                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=device_ids, output_device=output_device, **kwargs
                    )
            elif self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                # In case the model is already compiled using PyTorch 2.0 and the wrapped model in it
                # is a FSDP model, don't wrap it again
                is_type_fsdp = isinstance(model, FSDP) or (
                    is_compiled_module(model) and isinstance(model._orig_mod, FSDP)
                )

                if not is_type_fsdp:
                    self.state.fsdp_plugin.set_auto_wrap_policy(model)
                    fsdp_plugin = self.state.fsdp_plugin
                    kwargs = {
                        "sharding_strategy": fsdp_plugin.sharding_strategy,
                        "cpu_offload": fsdp_plugin.cpu_offload,
                        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                        "mixed_precision": fsdp_plugin.mixed_precision_policy,
                        "sync_module_states": fsdp_plugin.sync_module_states,
                        "backward_prefetch": fsdp_plugin.backward_prefetch,
                        "forward_prefetch": fsdp_plugin.forward_prefetch,
                        "use_orig_params": fsdp_plugin.use_orig_params,
                        "param_init_fn": fsdp_plugin.param_init_fn,
                        "ignored_modules": fsdp_plugin.ignored_modules,
                        "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                        "device_id": self.device,
                    }
                    model = FSDP(model, **kwargs)
                    if fsdp_plugin.activation_checkpointing:
                        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                            CheckpointImpl,
                            apply_activation_checkpointing,
                            checkpoint_wrapper,
                        )

                        apply_activation_checkpointing(
                            model,
                            checkpoint_wrapper_fn=functools.partial(
                                checkpoint_wrapper,
                                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                            ),
                            auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
                        )
                # if the previous and current models are same, delete the previous one
                if len(self._models) > 1 and (self._models[-2] is self._models[-1]):
                    del self._models[-2]
                self._models[-1] = model
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
            elif self.distributed_type == DistributedType.XLA and self.state.fork_launched:
                model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last and only if the model isn't already compiled.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO and not is_compiled_module(model):
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        result = [
            self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
            for obj in args
        ]

        if deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu"):
            if is_dataloader_present:
                batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
                if any(bs is None for bs in batch_sizes):
                    raise ValueError(
                        "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size. "
                        "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                        "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                    )
                if self.split_batches:
                    batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]

                batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
                if len(batch_sizes) > 1:
                    logger.info(
                        "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                        f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
                    )
            else:
                raise ValueError(
                    "When using DeepSpeed, `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "with `batch_size` attribute returning an integer value "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
        else:
            batch_size_per_device = deepspeed_plugin.get_value("train_micro_batch_size_per_gpu")

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.get_value("gradient_accumulation_steps")
            * self.num_processes,
            "gradient_clipping": 1.0,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        model = None
        optimizer = None
        scheduler = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                optimizer = obj
            elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
            ):
                scheduler = obj

        if optimizer is not None:
            if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot specify an optimizer in the config file and in the code at the same time. "
                    "Please remove the optimizer from the config file or "
                    "create `accelerate.utils.DummyOptim` in the code."
                )
            elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                )

            if isinstance(optimizer, (torch.optim.Optimizer)):
                deepspeed_plugin.deepspeed_config["zero_allow_untested_optimizer"] = True

        if scheduler is not None:
            if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot specify a scheduler in the config file and in the code at the same time. "
                    "Please remove the scheduler from the config file or "
                    "create `accelerate.utils.DummyScheduler` in the code."
                )
            elif (
                "scheduler" not in deepspeed_plugin.deepspeed_config
                and isinstance(scheduler, (DummyScheduler))
                and scheduler.lr_scheduler_callable is None
            ):
                raise ValueError(
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                )

        if optimizer is not None and scheduler is not None:
            if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
                    "`accelerate.utils.DummyOptim`."
                )

        if model is not None:
            # deal with config keys that use `auto` value and rely on model's hidden_size
            hidden_size_based_keys = [
                "zero_optimization.reduce_bucket_size",
                "zero_optimization.stage3_prefetch_bucket_size",
                "zero_optimization.stage3_param_persistence_threshold",
            ]
            hidden_size_auto_keys = [x for x in hidden_size_based_keys if deepspeed_plugin.is_auto(x)]
            if len(hidden_size_auto_keys) > 0:
                reasoning = (
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    + f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    + "`auto` values for these keys with an integer value of your choice."
                )
                if not hasattr(model, "config"):
                    raise ValueError("Can't find `model.config` entry, " + reasoning)

                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.hidden_sizes)
                else:
                    raise ValueError(
                        "Can find neither `model.config.hidden_size` nor `model.config.hidden_sizes`, " + reasoning
                    )

                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    }
                )

            if isinstance(optimizer, (DummyOptim)):
                config_kwargs.update(
                    {"optimizer.params.lr": optimizer.lr, "optimizer.params.weight_decay": optimizer.weight_decay}
                )
            if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is None:
                max_lr = (
                    getattr(scheduler.optimizer, "lr", None)
                    if getattr(scheduler.optimizer, "defaults", None) is None
                    else scheduler.optimizer.defaults["lr"]
                )
                config_kwargs.update(
                    {
                        "scheduler.params.warmup_min_lr": 0,
                        "scheduler.params.warmup_max_lr": max_lr,
                        "scheduler.params.warmup_num_steps": scheduler.warmup_num_steps,
                    }
                )
                if scheduler.total_num_steps is not None:
                    config_kwargs["scheduler.params.total_num_steps"] = (
                        math.ceil(scheduler.total_num_steps / self.num_processes)
                        if not self.split_batches
                        else scheduler.total_num_steps
                    )
            deepspeed_plugin.deepspeed_config_process(must_match=False, **config_kwargs)
            self.deepspeed_config = deepspeed_plugin.deepspeed_config
            kwargs = dict(model=model, config_params=self.deepspeed_config)
            if optimizer is not None:
                if isinstance(optimizer, (DummyOptim)):
                    kwargs["model_parameters"] = optimizer.params
                    if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is not None:
                        kwargs["lr_scheduler"] = scheduler.lr_scheduler_callable
                else:
                    if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                        "device", "none"
                    ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                        from deepspeed.ops.adam import DeepSpeedCPUAdam

                        defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                        optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                    kwargs["optimizer"] = optimizer
                    if scheduler is not None:
                        if type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES:
                            kwargs["lr_scheduler"] = scheduler

            engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
            if optimizer is not None:
                optimizer = DeepSpeedOptimizerWrapper(optimizer)
            if scheduler is not None:
                if lr_scheduler is None:
                    scheduler = AcceleratedScheduler(
                        scheduler,
                        optimizer,
                        step_with_optimizer=self.step_scheduler_with_optimizer,
                        split_batches=self.split_batches,
                    )
                else:
                    scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                    result[i] = optimizer
                elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                    type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    result[i] = scheduler
            # pointing for deepspeed_engine_wrapped.backward()
            self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
            self._models.append(engine)
            if optimizer is not None:
                self._optimizers.append(optimizer)
            if scheduler is not None:
                self._schedulers.append(scheduler)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                )
        return tuple(result)

    def _prepare_megatron_lm(self, *args):
        megatron_lm_plugin = self.state.megatron_lm_plugin
        if not megatron_lm_plugin.megatron_dataset_flag:
            batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
            if len(batch_sizes) == 0:
                raise ValueError(
                    "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using Megatron-LM."
                )

            micro_batch_size = min(batch_sizes) if megatron_lm_plugin.is_train_batch_min else max(batch_sizes)
            if len(batch_sizes) > 1:
                logger.info(
                    "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                    f"{megatron_lm_plugin.is_train_batch_min} will decide the `train_batch_size` ({micro_batch_size})."
                )
        else:
            for obj in args:
                if isinstance(obj, MegatronLMDummyDataLoader):
                    micro_batch_size = obj.dataset_args["micro_batch_size"]
                    break

        dp_degree = self.num_processes // (megatron_lm_plugin.tp_degree * megatron_lm_plugin.pp_degree)
        megatron_lm_plugin.set_training_args(micro_batch_size, dp_degree)

        model = None
        optimizer = None
        scheduler = None
        is_dummy_scheduler = False
        batch_data = None
        for obj in args:
            if isinstance(obj, torch.utils.data.DataLoader) and batch_data is None:
                batch_data = next(iter(obj))
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer)):
                optimizer = obj
            elif isinstance(obj, (LRScheduler, MegatronLMDummyScheduler)):
                scheduler = obj

        if model is not None:
            megatron_lm_plugin.set_network_size_args(model, batch_data)
        if optimizer is not None:
            megatron_lm_plugin.set_optimizer_type(optimizer)
        if scheduler is not None:
            is_dummy_scheduler = isinstance(scheduler, MegatronLMDummyScheduler)
            if not is_dummy_scheduler:
                raise ValueError(
                    "You can't use a custom scheduler with Megatron-LM. Please use the `accelerate.utils.MegatronLMDummyScheduler` instead."
                )
            megatron_lm_plugin.set_scheduler_args(scheduler)

        # initialize megatron-lm
        megatron_lm_initialize(self, args_defaults=megatron_lm_plugin.megatron_lm_default_args)
        counter = 0
        result = []
        for obj in args:
            if isinstance(obj, torch.utils.data.DataLoader):
                result.append(megatron_lm_prepare_data_loader(self, obj))
                counter += 1
            elif isinstance(obj, MegatronLMDummyDataLoader):
                if counter == 0:
                    obj.set_megatron_data_args()
                    dataloaders = megatron_lm_prepare_data_loader(self, obj)
                result.append(dataloaders[counter])
                counter += 1
            else:
                result.append(obj)

        if model is not None:
            model = megatron_lm_prepare_model(self)
        if optimizer is not None:
            optimizer = megatron_lm_prepare_optimizer(self, model)
        if scheduler is not None:
            scheduler = megatron_lm_prepare_scheduler(self, optimizer, scheduler)

        if model is not None:
            model = MegatronEngine(self, model, optimizer, scheduler)
        if optimizer is not None:
            optimizer = MegatronLMOptimizerWrapper(optimizer)
        if scheduler is not None:
            scheduler = MegatronLMSchedulerWrapper(scheduler, optimizer)

        for i in range(len(result)):
            if isinstance(result[i], torch.nn.Module):
                result[i] = model
            elif isinstance(result[i], torch.optim.Optimizer):
                result[i] = optimizer
            elif isinstance(result[i], MegatronLMDummyScheduler):
                result[i] = scheduler
        if model is not None:
            self._models.append(model)
        if optimizer is not None:
            self._optimizers.append(optimizer)
        if scheduler is not None:
            self._schedulers.append(scheduler)
        if len(self._models) > 1:
            raise AssertionError(
                "You can't use same `Accelerator()` instance with multiple models when using Megatron-LM"
            )
        return tuple(result)

    def _prepare_ipex(self, *args):
        if not is_ipex_available():
            raise ImportError(
                "IPEX is not installed or IPEX's version does not match current PyTorch version. Please refer"
                " to https://github.com/intel/intel-extension-for-pytorch."
            )
        else:
            import intel_extension_for_pytorch as ipex

        model = None
        optimizer = None
        result = [obj for obj in args]
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
                model.train()
            elif isinstance(obj, (torch.optim.Optimizer)):
                optimizer = obj
        if optimizer is not None and model is not None:
            dtype = torch.bfloat16 if self.state.mixed_precision == "bf16" else None
            if self.device.type == "xpu" and is_xpu_available():
                model = model.to(self.device)
                model, optimizer = torch.xpu.optimize(
                    model, optimizer=optimizer, dtype=dtype, inplace=True, level="O1"
                )
            else:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype, inplace=True, level="O1")
        for i in range(len(result)):
            if isinstance(result[i], torch.nn.Module):
                result[i] = model
            elif isinstance(result[i], (torch.optim.Optimizer)):
                result[i] = optimizer
        return tuple(result)

    def _prepare_msamp(self, *args):
        if not is_msamp_available():
            raise ImportError(
                "MS-AMP was not found on your system. Please ensure that MS-AMP is available "
                " or choose `'te'` as the backend for FP8 mixed precision training."
            )
        else:
            import msamp

        model, optimizer = None, None
        num_models, num_optimizers = 0, 0
        result = [obj for obj in args]
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
                num_models += 1
            elif isinstance(obj, (torch.optim.Optimizer)):
                optimizer = obj
                num_optimizers += 1
        if optimizer is None or model is None:
            raise ValueError(
                "You must pass a model and an optimizer together to `accelerate.prepare()` when using MS-AMP."
            )
        elif num_models > 1 or num_optimizers > 1:
            raise ValueError(
                f"You can't use multiple models ({num_models}) or optimizers {num_optimizers} with MS-AMP."
            )
        else:
            model, optimizer = msamp.initialize(model, optimizer, opt_level=self.fp8_recipe_handler.opt_level)
        for i in range(len(result)):
            if isinstance(result[i], torch.nn.Module):
                result[i] = model
            elif isinstance(result[i], (torch.optim.Optimizer)):
                result[i] = optimizer
        return tuple(result)

    def prepare_data_loader(
        self, data_loader: torch.utils.data.DataLoader, device_placement=None, slice_fn_for_dispatch=None
    ):
        """
        Prepares a PyTorch DataLoader for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            data_loader (`torch.utils.data.DataLoader`):
                A vanilla PyTorch DataLoader to prepare
            device_placement (`bool`, *optional*):
                Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
                `self.device_placement`.
            slice_fn_for_dispatch (`Callable`, *optional*`):
                If passed, this function will be used to slice tensors across `num_processes`. Will default to
                [`~utils.slice_tensors`]. This argument is used only when `dispatch_batches` is set to `True` and will
                be ignored otherwise.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
        ```
        """
        # Ensure we can't double wrap a DataLoader due to `find_batch_size`
        if getattr(data_loader, "_is_accelerate_prepared", False):
            if data_loader not in self._dataloaders:
                self._dataloaders.append(data_loader)
            return data_loader
        if device_placement is None:
            device_placement = self.device_placement if self.distributed_type != DistributedType.XLA else False
        prepared_data_loader = prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
            even_batches=self.even_batches,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            use_seedable_sampler=self.use_seedable_sampler,
        )
        self._dataloaders.append(prepared_data_loader)
        return prepared_data_loader

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer, device_placement=None):
        """
        Prepares a PyTorch Optimizer for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            optimizer (`torch.optim.Optimizer`):
                A vanilla PyTorch optimizer to prepare
            device_placement (`bool`, *optional*):
                Whether or not to place the optimizer on the proper device. Will default to `self.device_placement`.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> optimizer = torch.optim.Adam(...)
        >>> optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
        ```
        """
        # Ensure we can't double wrap an optimizer due to `find_batch_size`
        if getattr(optimizer, "_is_accelerate_prepared", False):
            if optimizer not in self._optimizers:
                self._optimizers.append(optimizer)
            return optimizer
        if device_placement is None:
            device_placement = self.device_placement
        optimizer = AcceleratedOptimizer(optimizer, device_placement=device_placement, scaler=self.scaler)
        self._optimizers.append(optimizer)
        return optimizer

    def prepare_scheduler(self, scheduler: LRScheduler):
        """
        Prepares a PyTorch Scheduler for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            scheduler (`torch.optim.lr_scheduler.LRScheduler`):
                A vanilla PyTorch scheduler to prepare

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> optimizer = torch.optim.Adam(...)
        >>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
        >>> scheduler = accelerator.prepare_scheduler(scheduler)
        ```
        """
        # Ensure we can't double wrap a scheduler due to `find_batch_size`
        if getattr(scheduler, "_is_accelerate_prepared", False):
            if scheduler not in self._schedulers:
                self._schedulers.append(scheduler)
            return scheduler
        # We try to find the optimizer associated with `scheduler`, the default is the full list.
        optimizer = self._optimizers
        for opt in self._optimizers:
            if getattr(scheduler, "optimizer", None) == opt.optimizer:
                optimizer = opt
                break
        scheduler = AcceleratedScheduler(
            scheduler,
            optimizer,
            step_with_optimizer=self.step_scheduler_with_optimizer,
            split_batches=self.split_batches,
        )
        self._schedulers.append(scheduler)
        return scheduler

    def backward(self, loss, **kwargs):
        """
        Scales the gradients in accordance to the `GradientAccumulationPlugin` and calls the correct `backward()` based
        on the configuration.

        Should be used in lieu of `loss.backward()`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        ```
        """
        if self.distributed_type != DistributedType.DEEPSPEED:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.gradient_accumulation_steps
        if self.distributed_type == DistributedType.DEEPSPEED:
            self.deepspeed_engine_wrapped.backward(loss, **kwargs)
        elif self.distributed_type == DistributedType.MEGATRON_LM:
            return
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def set_trigger(self):
        """
        Sets the internal trigger tensor to 1 on the current process. A latter check should follow using this which
        will check across all processes.

        Note:
            Does not require `wait_for_everyone()`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume later in the training script
        >>> # `should_do_breakpoint` is a custom function to monitor when to break,
        >>> # e.g. when the loss is NaN
        >>> if should_do_breakpoint(loss):
        ...     accelerator.set_trigger()
        >>> # Assume later in the training script
        >>> if accelerator.check_breakpoint():
        ...     break
        ```
        """
        self.flag_tensor = torch.tensor(1, device=self.device)

    def check_trigger(self):
        """
        Checks if the internal trigger tensor has been set to 1 in any of the processes. If so, will return `True` and
        reset the trigger tensor to 0.

        Note:
            Does not require `wait_for_everyone()`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume later in the training script
        >>> # `should_do_breakpoint` is a custom function to monitor when to break,
        >>> # e.g. when the loss is NaN
        >>> if should_do_breakpoint(loss):
        ...     accelerator.set_trigger()
        >>> # Assume later in the training script
        >>> if accelerator.check_trigger():
        ...     break
        ```
        """
        # Now that we are outside `__init__`, we can initialize it if it is `None` on device
        if self.flag_tensor is None:
            self.flag_tensor = torch.tensor(0, device=self.device)
        flag_tensor = self.reduce(self.flag_tensor)
        if flag_tensor.item() >= 1:
            self.flag_tensor = torch.tensor(0, device=self.device)
            return True
        return False

    def unscale_gradients(self, optimizer=None):
        """
        Unscale the gradients in mixed precision training with AMP. This is a noop in all other settings.

        Likely should be called through [`Accelerator.clip_grad_norm_`] or [`Accelerator.clip_grad_value_`]

        Args:
            optimizer (`torch.optim.Optimizer` or `list[torch.optim.Optimizer]`, *optional*):
                The optimizer(s) for which to unscale gradients. If not set, will unscale gradients on all optimizers
                that were passed to [`~Accelerator.prepare`].

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer = accelerator.prepare(model, optimizer)
        >>> outputs = model(inputs)
        >>> loss = loss_fn(outputs, labels)
        >>> accelerator.backward(loss)
        >>> accelerator.unscale_gradients(optimizer=optimizer)
        ```
        """
        if self.native_amp and self.mixed_precision == "fp16":
            if optimizer is None:
                # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
                optimizer = self._optimizers
            elif not isinstance(optimizer, (tuple, list)):
                optimizer = [optimizer]
            for opt in optimizer:
                while isinstance(opt, AcceleratedOptimizer):
                    opt = opt.optimizer
                self.scaler.unscale_(opt)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """
        Should be used in place of `torch.nn.utils.clip_grad_norm_`.

        Returns:
            `torch.Tensor`: Total norm of the parameter gradients (viewed as a single vector).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        ...     optimizer.step()
        ```
        """
        if self.distributed_type == DistributedType.FSDP:
            self.unscale_gradients()
            parameters = [p for p in parameters]
            for model in self._models:
                if parameters == [p for p in model.parameters()]:
                    return model.clip_grad_norm_(max_norm, norm_type)
        elif self.distributed_type == DistributedType.DEEPSPEED:
            # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
            # We cannot return the gradient norm because DeepSpeed does it.
            return None
        elif self.distributed_type == DistributedType.XLA:
            # Reduce gradients first for XLA
            for acc_opt in self._optimizers:
                if not acc_opt.gradient_state.is_xla_gradients_synced:
                    opt = acc_opt
                    while isinstance(opt, AcceleratedOptimizer):
                        opt = opt.optimizer
                    gradients = xm._fetch_gradients(opt)
                    # Use xm.all_reduce to perform an in-place all-reduce. Recusrsive all-reduce each tensor
                    # one by one in self.reduce is non-inplace.
                    xm.all_reduce("sum", gradients, scale=1.0 / self.num_processes)
                    # Set is_xla_gradients_synced to True to avoid all-reduce twice in the AcceleratedOptimizer step.
                    acc_opt.gradient_state.is_xla_gradients_synced = True
        self.unscale_gradients()
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of `torch.nn.utils.clip_grad_value_`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_value_(model.parameters(), clip_value)
        ...     optimizer.step()
        ```
        """
        if self.distributed_type in [DistributedType.DEEPSPEED, DistributedType.FSDP]:
            raise Exception("DeepSpeed and FSDP  do not support `clip_grad_value_`. Use `clip_grad_norm_` instead.")
        self.unscale_gradients()
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    def gather(self, tensor):
        """
        Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
        regroup the predictions from all processes when doing evaluation.

        Note:
            This gather happens in all processes.

        Args:
            tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
                The tensors to gather across all processes.

        Returns:
            `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`: The gathered tensor(s). Note that the
            first dimension of the result is *num_processes* multiplied by the first dimension of the input tensors.

        Example:

        ```python
        >>> # Assuming four processes
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.tensor([accelerator.process_index])
        >>> gathered_tensor = accelerator.gather(process_tensor)
        >>> gathered_tensor
        tensor([0, 1, 2, 3])
        ```
        """
        return gather(tensor)

    def gather_for_metrics(self, input_data):
        """
        Gathers `input_data` and potentially drops duplicates in the last batch if on a distributed system. Should be
        used for gathering the inputs and targets for metric calculation.

        Args:
            input (`torch.Tensor`, `object`, a nested tuple/list/dictionary of `torch.Tensor`, or a nested tuple/list/dictionary of `object`):
                The tensors or objects for calculating metrics across all processes

        Example:

        ```python
        >>> # Assuming two processes, with a batch size of 5 on a dataset with 9 samples
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader = torch.utils.data.DataLoader(range(9), batch_size=5)
        >>> dataloader = accelerator.prepare(dataloader)
        >>> batch = next(iter(dataloader))
        >>> gathered_items = accelerator.gather_for_metrics(batch)
        >>> len(gathered_items)
        9
        ```
        """

        try:
            recursively_apply(lambda x: x, input_data, error_on_other_type=True)
            all_tensors = True
        except TypeError:
            all_tensors = False

        if not all_tensors:
            data = gather_object(input_data)
        else:
            data = self.gather(input_data)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                    )
                    return data
                elif self.gradient_state.remainder > 0:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder]

                    return recursively_apply(_adjust_samples, data)
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return data
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return data
        except Exception:
            # Dataset had no length or raised an error
            return data

    def reduce(self, tensor, reduction="sum", scale=1.0):
        """
        Reduce the values in *tensor* across all processes based on *reduction*.

        Note:
            All processes get the reduced value.

        Args:
            tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
                The tensors to reduce across all processes.
            reduction (`str`, *optional*, defaults to "sum"):
                A reduction type, can be one of 'sum', 'mean', or 'none'. If 'none', will not perform any operation.
            scale (`float`, *optional*, defaults to 1.0):
                A default scaling value to be applied after the reduce, only valied on XLA.

        Returns:
            `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`:
                The reduced tensor(s).

        Example:

        ```python
        >>> # Assuming two processes
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.arange(accelerator.num_processes) + 1 + (2 * accelerator.process_index)
        >>> process_tensor = process_tensor.to(accelerator.device)
        >>> reduced_tensor = accelerator.reduce(process_tensor, reduction="sum")
        >>> reduced_tensor
        tensor([4, 6])
        ```
        """
        return reduce(tensor, reduction, scale)

    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.

        Args:
            tensor (nested list/tuple/dictionary of `torch.Tensor`):
                The data to gather.
            dim (`int`, *optional*, defaults to 0):
                The dimension on which to pad.
            pad_index (`int`, *optional*, defaults to 0):
                The value with which to pad.
            pad_first (`bool`, *optional*, defaults to `False`):
                Whether to pad at the beginning or the end.

        Returns:
            `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`:
                The padded tensor(s).

        Example:

        ```python
        >>> # Assuming two processes, with the first processes having a tensor of size 1 and the second of size 2
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.arange(accelerator.process_index + 1).to(accelerator.device)
        >>> padded_tensor = accelerator.pad_across_processes(process_tensor)
        >>> padded_tensor.shape
        torch.Size([2])
        ```
        """
        return pad_across_processes(tensor, dim=dim, pad_index=pad_index, pad_first=pad_first)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        """
        Unwraps the `model` from the additional layer possible added by [`~Accelerator.prepare`]. Useful before saving
        the model.

        Args:
            model (`torch.nn.Module`):
                The model to unwrap.
            keep_fp32_wrapper (`bool`, *optional*, defaults to `True`):
                Whether to not remove the mixed precision hook if it was added.

        Returns:
            `torch.nn.Module`: The unwrapped model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> from torch.nn.parallel import DistributedDataParallel
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model = accelerator.prepare(MyModel())
        >>> print(model.__class__.__name__)
        DistributedDataParallel

        >>> model = accelerator.unwrap_model(model)
        >>> print(model.__class__.__name__)
        MyModel
        ```
        """
        return extract_model_from_parallel(model, keep_fp32_wrapper)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> if accelerator.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> accelerator.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        wait_for_everyone()

    @on_main_process
    def init_trackers(self, project_name: str, config: dict | None = None, init_kwargs: dict | None = {}):
        """
        Initializes a run for all trackers stored in `self.log_with`, potentially with starting configurations

        Args:
            project_name (`str`):
                The name of the project. All trackers will save their data based on this
            config (`dict`, *optional*):
                Optional starting configuration to be logged.
            init_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `__init__` function. Should be
                formatted like so:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers(
        ...     project_name="my_project",
        ...     config={"learning_rate": 0.001, "batch_size": 32},
        ...     init_kwargs={"tensorboard": {"flush_secs": 60}},
        ... )
        ```
        """
        for tracker in self.log_with:
            if issubclass(type(tracker), GeneralTracker):
                # Custom trackers are already initialized
                self.trackers.append(tracker)
            else:
                tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
                if tracker_init.requires_logging_directory:
                    # We can skip this check since it was done in `__init__`
                    self.trackers.append(
                        tracker_init(project_name, self.logging_dir, **init_kwargs.get(str(tracker), {}))
                    )
                else:
                    self.trackers.append(tracker_init(project_name, **init_kwargs.get(str(tracker), {})))
        if config is not None:
            for tracker in self.trackers:
                tracker.store_init_configuration(config)

    def get_tracker(self, name: str, unwrap: bool = False):
        """
        Returns a `tracker` from `self.trackers` based on `name` on the main process only.

        Args:
            name (`str`):
                The name of a tracker, corresponding to the `.name` property.
            unwrap (`bool`):
                Whether to return the internal tracking mechanism or to return the wrapped tracker instead
                (recommended).

        Returns:
            `GeneralTracker`: The tracker corresponding to `name` if it exists.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> tensorboard_tracker = accelerator.get_tracker("tensorboard")
        ```
        """
        if len(self.trackers) > 0:
            for tracker in self.trackers:
                if tracker.name == name:
                    return tracker.tracker if unwrap else tracker
            raise ValueError(f"{name} is not an available tracker stored inside the `Accelerator`.")
        # Handle tracker only made on main process
        return GeneralTracker(_blank=True)

    @on_main_process
    def log(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        """
        Logs `values` to all stored trackers in `self.trackers` on the main process only.

        Args:
            values (`dict`):
                Values should be a dictionary-like object containing only types `int`, `float`, or `str`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            log_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `log` function. Should be formatted
                like so:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> accelerator.log({"loss": 0.5, "accuracy": 0.9})
        ```
        """
        for tracker in self.trackers:
            tracker.log(values, step=step, **log_kwargs.get(tracker.name, {}))

    @on_main_process
    def end_training(self):
        """
        Runs any special end training behaviors, such as stopping trackers on the main process only. Should always be
        called at the end of your script if using experiment tracking.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> # Do training
        >>> accelerator.end_training()
        ```
        """
        for tracker in self.trackers:
            tracker.finish()

    def save(self, obj, f, safe_serialization=False):
        """
        Save the object passed to disk once per machine. Use in place of `torch.save`.

        Args:
            obj (`object`): The object to save.
            f (`str` or `os.PathLike`): Where to save the content of `obj`.
            safe_serialization (`bool`, *optional*, defaults to `False`): Whether to save `obj` using `safetensors`

        Note:
            If `save_on_each_node` was passed in as a `ProjectConfiguration`, will save the object once per node,
            rather than only once on the main node.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> arr = [0, 1, 2, 3]
        >>> accelerator.save(arr, "array.pkl")
        ```
        """
        save(
            obj,
            f,
            save_on_each_node=self.project_configuration.save_on_each_node,
            safe_serialization=safe_serialization,
        )

    def save_model(
        self,
        model: torch.nn.Module,
        save_directory: Union[str, os.PathLike],
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = True,
    ):
        """
        Save a model so that it can be re-loaded using load_checkpoint_in_model

        Arguments:
            model: (`torch.nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model = ...
        >>> accelerator.save_model(model, save_directory)
        ```
        """

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # get the state_dict of the model
        if any(
            [
                module._hf_hook.offload
                for module in model.modules()
                if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, AlignDevicesHook)
            ]
        ):
            state_dict = get_state_dict_offloaded_model(model)
        else:
            if any(param.device == torch.device("meta") for param in model.parameters()):
                raise RuntimeError("You can't save the model since some parameters are on the meta device.")
            state_dict = self.get_state_dict(model)

        if safe_serialization:
            state_dict = clean_state_dict_for_safetensors(state_dict)
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

        # Shard the model if it is too big.
        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            self.save(shard, os.path.join(save_directory, shard_file), safe_serialization=safe_serialization)

        if index is None:
            path_to_weights = os.path.join(save_directory, WEIGHTS_NAME)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def register_save_state_pre_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """
        Registers a pre hook to be run before `save_checkpoint` is called in [`Accelerator.save_state`].

        Args:
            hook (`Callable`):
                A function to be called in [`Accelerator.save_state`] before `save_checkpoint`.

        The hook should have the following signature:

        `hook(models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], input_dir: str) -> None`

        The `models` argument are the models as saved in the accelerator state under `accelerator._models`, `weigths`
        argument are the state dicts of the `models`, and the `input_dir` argument is the `input_dir` argument passed
        to [`Accelerator.load_state`].

        <Tip>

        Should only be used in conjunction with [`Accelerator.register_load_state_pre_hook`]. Can be useful to save
        configurations in addition to model weights. Can also be used to overwrite model saving with a customized
        method. In this case, make sure to remove already loaded weights from the weights list.

        </Tip>

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling
            `handle.remove()`
        """
        handle = hooks.RemovableHandle(self._save_model_state_pre_hook)
        self._save_model_state_pre_hook[handle.id] = hook
        return handle

    def save_state(self, output_dir: str = None, safe_serialization: bool = True, **save_model_func_kwargs):
        """
        Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.

        If a `ProjectConfiguration` was passed to the `Accelerator` object with `automatic_checkpoint_naming` enabled
        then checkpoints will be saved to `self.project_dir/checkpoints`. If the number of current saves is greater
        than `total_limit` then the oldest save is deleted. Each checkpoint is saved in seperate folders named
        `checkpoint_<iteration>`.

        Otherwise they are just saved to `output_dir`.

        <Tip>

        Should only be used when wanting to save a checkpoint during training and restoring the state in the same
        environment.

        </Tip>

        Args:
            output_dir (`str` or `os.PathLike`):
                The name of the folder to save all relevant weights and states.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            save_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for saving model which can be passed to the underlying save function, such
                as optional arguments for DeepSpeed's `save_checkpoint` function.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.save_state(output_dir="my_checkpoint")
        ```
        """
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            if (
                self.project_configuration.total_limit is not None
                and (len(folders) + 1 > self.project_configuration.total_limit)
                and self.is_main_process
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

                folders.sort(key=_inner)
                logger.warning(
                    f"Deleting {len(folders) + 1 - self.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - self.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = os.path.join(output_dir, f"checkpoint_{self.save_iteration}")
            if os.path.exists(output_dir):
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({self.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
            self.wait_for_everyone()
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        if self.distributed_type == DistributedType.XLA:
            # Finish running the previous step before checkpointing
            xm.mark_step()

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Saving FSDP model")
                save_fsdp_model(self.state.fsdp_plugin, self, model, output_dir, i)
                logger.info(f"FSDP Model saved to output dir {output_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Saving DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.save_checkpoint(output_dir, ckpt_id, **save_model_func_kwargs)
                logger.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, ckpt_id)}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                logger.info("Saving Megatron-LM Model, Optimizer and Scheduler")
                model.save_checkpoint(output_dir)
                logger.info(f"Megatron-LM Model , Optimizer and Scheduler saved to output dir {output_dir}")
            else:
                weights.append(self.get_state_dict(model, unwrap=False))

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(self._optimizers):
                logger.info("Saving FSDP Optimizer")
                save_fsdp_optimizer(self.state.fsdp_plugin, self, opt, self._models[i], output_dir, i)
                logger.info(f"FSDP Optimizer saved to output dir {output_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        # Save the samplers of the dataloaders
        dataloaders = self._dataloaders

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir,
            weights,
            optimizers,
            schedulers,
            dataloaders,
            self.state.process_index,
            self.scaler,
            save_on_each_node=self.project_configuration.save_on_each_node,
            safe_serialization=safe_serialization,
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i, save_on_each_node=self.project_configuration.save_on_each_node)
        self.project_configuration.iteration += 1
        return save_location

    def register_load_state_pre_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """
        Registers a pre hook to be run before [`load_checkpoint`] is called in [`Accelerator.load_state`].

        Args:
            hook (`Callable`):
                A function to be called in [`Accelerator.load_state`] before `load_checkpoint`.

        The hook should have the following signature:

        `hook(models: list[torch.nn.Module], input_dir: str) -> None`

        The `models` argument are the models as saved in the accelerator state under `accelerator._models`, and the
        `input_dir` argument is the `input_dir` argument passed to [`Accelerator.load_state`].

        <Tip>

        Should only be used in conjunction with [`Accelerator.register_save_state_pre_hook`]. Can be useful to load
        configurations in addition to model weights. Can also be used to overwrite model loading with a customized
        method. In this case, make sure to remove already loaded models from the models list.

        </Tip>

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling
            `handle.remove()`
        """
        handle = hooks.RemovableHandle(self._load_model_state_pre_hook)
        self._load_model_state_pre_hook[handle.id] = hook
        return handle

    def load_state(self, input_dir: str = None, **load_model_func_kwargs):
        """
        Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.

        <Tip>

        Should only be used in conjunction with [`Accelerator.save_state`]. If a file is not registered for
        checkpointing, it will not be loaded if stored in the directory.

        </Tip>

        Args:
            input_dir (`str` or `os.PathLike`):
                The name of the folder all relevant weights and states were saved in. Can be `None` if
                `automatic_checkpoint_naming` is used, and will pick up from the latest checkpoint.
            load_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for loading model which can be passed to the underlying load function,
                such as optional arguments for DeepSpeed's `load_checkpoint` function or a `map_location` to load the
                model and optimizer on.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.load_state("my_checkpoint")
        ```
        """
        if input_dir is not None:
            # Check if folder exists
            input_dir = os.path.expanduser(input_dir)
            if not os.path.isdir(input_dir):
                raise ValueError(f"Tried to find {input_dir} but folder does not exist")
        elif self.project_configuration.automatic_checkpoint_naming:
            # Pick up from automatic checkpoint naming
            input_dir = os.path.join(self.project_dir, "checkpoints")
            folders = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]

            def _inner(folder):
                return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

            folders.sort(key=_inner)
            input_dir = folders[-1]
        else:
            raise ValueError("No input_dir provided and automatic checkpoint naming is disabled.")
        logger.info(f"Loading states from {input_dir}")

        # Load the models taking care of FSDP and DeepSpeed nuances
        models = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Loading FSDP model")
                load_fsdp_model(self.state.fsdp_plugin, self, model, input_dir, i)
                logger.info(f"FSDP Model loaded from input dir {input_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Loading DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.load_checkpoint(input_dir, ckpt_id, **load_model_func_kwargs)
                logger.info(f"DeepSpeed Model and Optimizer loaded from input dir {os.path.join(input_dir, ckpt_id)}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                logger.info("Loading Megatron-LM Model, Optimizer and Scheduler")
                model.load_checkpoint(input_dir)
                logger.info(f"Megatron-LM Model , Optimizer and Scheduler loaded from input dir {input_dir}")
            else:
                models.append(model)

        # Load the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(self._optimizers):
                logger.info("Loading FSDP Optimizer")
                load_fsdp_optimizer(self.state.fsdp_plugin, self, opt, self._models[i], input_dir, i)
                logger.info(f"FSDP Optimizer loaded from input dir {input_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Load the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        dataloaders = self._dataloaders

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._load_model_state_pre_hook.values():
            hook(models, input_dir)

        map_location = load_model_func_kwargs.pop("map_location", None)
        if map_location is None:
            if self.num_processes > 1 and self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
            ):
                map_location = "on_device"
            else:
                map_location = "cpu"

        load_accelerator_state(
            input_dir,
            models,
            optimizers,
            schedulers,
            dataloaders,
            self.state.process_index,
            self.scaler,
            map_location,
            **load_model_func_kwargs,
        )
        custom_checkpoints = [
            f for f in os.listdir(input_dir) if re.search(r"^custom_checkpoint_\d+\.pkl$", f) is not None
        ]
        if len(custom_checkpoints) != len(self._custom_objects):
            err = "Number of custom checkpoints in folder {input_dir} does not match the number of registered objects:"
            err += f"\n\tFound checkpoints: {len(custom_checkpoints)}"
            err += f"\n\tRegistered objects: {len(self._custom_objects)}\n"
            err += "Please make sure to only load checkpoints from folders that were created with the same set of registered objects,"
            err += "or avoid using `custom_checkpoint` in the filename for files in that same directory and load them in manually."
            raise RuntimeError(err)
        else:
            logger.info(f"Loading in {len(custom_checkpoints)} custom states")
            for index, obj in enumerate(self._custom_objects):
                load_custom_state(obj, input_dir, index)

    def free_memory(self):
        """
        Will release all references to the internal objects stored and call the garbage collector. You should call this
        method between two trainings with different models/optimizers. Also will reset `Accelerator.step` to 0.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, scheduler = ...
        >>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        >>> accelerator.free_memory()
        >>> del model, optimizer, scheduler
        ```
        """
        self._schedulers = []
        self._optimizers = []
        self._models = []
        self._dataloaders = []
        self.deepspeed_engine_wrapped = None
        self.step = 0
        release_memory()

    def clear(self):
        """
        Alias for [`Accelerate.free_memory`], releases all references to the internal objects stored and call the
        garbage collector. You should call this method between two trainings with different models/optimizers.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, scheduler = ...
        >>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        >>> accelerator.free_memory()
        >>> del model, optimizer, scheduler
        ```
        """
        self.free_memory()

    def _get_named_parameters(self, *args):
        named_parameters = {}
        for obj in args:
            if isinstance(obj, torch.nn.Module):
                obj = extract_model_from_parallel(obj)
                named_parameters.update({n: p for n, p in obj.named_parameters()})
        return named_parameters

    def _get_devices(self, *args):
        model_device = None
        optimizer_device = None
        for obj in args:
            # Loop through model parameters and stop at the first once we have its device.
            if isinstance(obj, torch.nn.Module):
                for param in obj.parameters():
                    model_device = param.device
                    break
            # Loop through optimizer parameters groups and stop at the first once we have its device.
            if isinstance(obj, torch.optim.Optimizer):
                for param_group in obj.param_groups:
                    if len(param_group["params"]) > 0:
                        optimizer_device = param_group["params"][0].device
                        break
        return (model_device, optimizer_device)

    def get_state_dict(self, model, unwrap=True):
        """
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict

        Returns:
            `dict`: The state dictionary of the model potentially without full precision.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> net = torch.nn.Linear(2, 2)
        >>> net = accelerator.prepare(net)
        >>> state_dict = accelerator.get_state_dict(net)
        ```
        """

        if self.distributed_type == DistributedType.DEEPSPEED:
            if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                if model.zero_gather_16bit_weights_on_model_save():
                    state_dict = model._zero3_consolidated_16bit_state_dict()
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self.unwrap_model(model).state_dict())
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()

        return state_dict

    def register_for_checkpointing(self, *objects):
        """
        Makes note of `objects` and will save or load them in during `save_state` or `load_state`.

        These should be utilized when the state is being loaded or saved in the same script. It is not designed to be
        used in different scripts.

        <Tip>

        Every `object` must have a `load_state_dict` and `state_dict` function to be stored.

        </Tip>

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume `CustomObject` has a `state_dict` and `load_state_dict` function.
        >>> obj = CustomObject()
        >>> accelerator.register_for_checkpointing(obj)
        >>> accelerator.save_state("checkpoint.pt")
        ```
        """
        invalid_objects = []
        for obj in objects:
            if not hasattr(obj, "state_dict") or not hasattr(obj, "load_state_dict"):
                invalid_objects.append(obj)
        if len(invalid_objects) > 0:
            err = "All `objects` must include a `state_dict` and `load_state_dict` function to be stored. The following inputs are invalid:"
            for index, obj in enumerate(invalid_objects):
                err += f"\n\t- Item at index {index}, `{get_pretty_name(obj)}`"
            raise ValueError(err)
        self._custom_objects.extend(objects)

    @contextmanager
    def autocast(self, cache_enabled: bool = False, autocast_handler: AutocastKwargs = None):
        """
        Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
        different will happen otherwise.

        A different `autocast_handler` can be passed in to override the one set in the `Accelerator` object. This is
        useful in blocks under `autocast` where you want to revert to fp32.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(mixed_precision="fp16")
        >>> with accelerator.autocast():
        ...     train()
        ```
        """
        if cache_enabled:
            warnings.warn(
                "Passing `cache_enabled=True` to `accelerator.autocast` is deprecated and will be removed in v0.23.0. "
                "Please use the `AutocastKwargs` class instead and pass it to the `Accelerator` as a `kwarg_handler`.",
                FutureWarning,
            )
            if self.autocast_handler is not None:
                self.autocast_handler.cache_enabled = True
            else:
                self.autocast_handler = AutocastKwargs(cache_enabled=True)
        if autocast_handler is None:
            autocast_handler = self.autocast_handler
        autocast_context = get_mixed_precision_context_manager(self.native_amp, autocast_handler)
        autocast_context.__enter__()
        # TODO: should the `yield` be in a try/finally block?
        yield
        autocast_context.__exit__(*sys.exc_info())

    @property
    def optimizer_step_was_skipped(self):
        """
        Whether or not the optimizer update was skipped (because of gradient overflow in mixed precision), in which
        case the learning rate should not be changed.
        """
        for optimizer in self._optimizers:
            if optimizer.step_was_skipped:
                return True
        return False

    def skip_first_batches(self, dataloader, num_batches: int = 0):
        """
        Creates a new `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.

        Args:
            dataloader (`torch.utils.data.DataLoader`): The data loader in which to skip batches.
            num_batches (`int`, *optional*, defaults to 0): The number of batches to skip

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
        >>> skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=2)
        >>> # for the first epoch only
        >>> for input, target in skipped_dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     optimizer.step()

        >>> # subsequent epochs
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     ...
        ```
        """
        return skip_first_batches(dataloader, num_batches=num_batches)

    def __deepcopy__(self, memo):
        logger.info("Deep copying the `Accelerator` object, note that this will point to the same original object.")
        return self

    def verify_device_map(self, model: torch.nn.Module) -> bool:
        """
        Verifies that `model` has not been prepared with big model inference with a device-map resembling `auto`.
        """
        # Checks if any of the child modules has the attribute `hf_device_map` and this map has more than one entry.
        for m in model.modules():
            if hasattr(m, "hf_device_map") and len(m.hf_device_map) > 1:
                return True

        return False
