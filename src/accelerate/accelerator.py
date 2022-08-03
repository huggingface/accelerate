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

import contextlib
import gc
import math
import os
import sys
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import List, Optional, Union

import torch

from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import prepare_data_loader
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, parse_flag_from_env
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
    MODEL_NAME,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    FullyShardedDataParallelPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    PrecisionType,
    RNGType,
    compare_versions,
    convert_outputs_to_fp32,
    extract_model_from_parallel,
    gather,
    get_pretty_name,
    is_bf16_available,
    is_deepspeed_available,
    is_torch_version,
    is_tpu_available,
    pad_across_processes,
    recursively_apply,
    reduce,
    save,
    wait_for_everyone,
)


if is_deepspeed_available():
    import deepspeed

    from .utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

if is_tpu_available(check_device=False):
    import torch_xla.distributed.xla_multiprocessing as xmp

logger = get_logger(__name__)


class Accelerator:
    """
    Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

    Args:
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
            etc...).
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        mixed_precision (`str`, *optional*):
            Whether or not to use mixed precision training (fp16 or bfloat16). Choose from 'no','fp16','bf16'. Will
            default to the value in the environment variable `MIXED_PRECISION`, which will use the default value in the
            accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp16'
            requires pytorch 1.6 or higher. 'bf16' requires pytorch 1.10 or higher.
        gradient_accumulation_steps (`int`, *optional*, default to 1):
            The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
            `Accelerator.accumulate`.
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
            the execution on one process only.
        deepspeed_plugin (`DeepSpeedPlugin`, *optional*):
            Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        fsdp_plugin (`FullyShardedDataParallelPlugin`, *optional*):
            Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
            using *accelerate config*
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
            If `"all`" is selected, will pick up all available trackers in the environment and intialize them. Can also
            accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        logging_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing logs of locally-compatible loggers.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        step_scheduler_with_optimizer (`bool`, *optional`, defaults to `True`):
            Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
            done under certain circumstances (at the end of each epoch, for instance).
        kwargs_handlers (`List[KwargHandler]`, *optional*)
            A list of `KwargHandler` to customize how the objects related to distributed training or mixed precision
            are created. See [kwargs](kwargs) for more information.

    **Attributes:**

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
        split_batches: bool = False,
        fp16: bool = None,
        mixed_precision: Union[PrecisionType, str] = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        deepspeed_plugin: DeepSpeedPlugin = None,
        fsdp_plugin: FullyShardedDataParallelPlugin = None,
        rng_types: Optional[List[Union[str, RNGType]]] = None,
        log_with: Optional[List[Union[str, LoggerType, GeneralTracker]]] = None,
        logging_dir: Optional[Union[str, os.PathLike]] = None,
        dispatch_batches: Optional[bool] = None,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: Optional[List[KwargsHandler]] = None,
    ):
        self.logging_dir = logging_dir
        trackers = filter_trackers(log_with, self.logging_dir)
        if len(trackers) < 1 and log_with is not None:
            warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
        self.log_with = trackers

        if mixed_precision is not None:
            mixed_precision = str(mixed_precision)
            if mixed_precision not in PrecisionType:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
                )

        if fp16:
            warnings.warn('fp16=True is deprecated. Use mixed_precision="fp16" instead.', DeprecationWarning)
            mixed_precision = "fp16"

        if deepspeed_plugin is None:  # init from env variables
            deepspeed_plugin = DeepSpeedPlugin() if os.environ.get("USE_DEEPSPEED", "false") == "true" else None
        else:
            assert isinstance(
                deepspeed_plugin, DeepSpeedPlugin
            ), "`deepspeed_plugin` must be a DeepSpeedPlugin object."
            os.environ["USE_DEEPSPEED"] = "true"  # use DeepSpeed if plugin is provided
        if deepspeed_plugin:
            if not is_deepspeed_available():
                raise ImportError("DeepSpeed is not installed => run `pip install deepspeed` or build it from source.")
            if compare_versions("deepspeed", "<", "0.6.5"):
                raise ImportError("DeepSpeed version must be >= 0.6.5. Please update DeepSpeed.")

            mixed_precision = os.environ.get("MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
            deepspeed_plugin.set_mixed_precision(mixed_precision)
            deepspeed_plugin.set_deepspeed_weakref()

        if os.environ.get("USE_FSDP", "false") == "true" or isinstance(fsdp_plugin, FullyShardedDataParallelPlugin):
            if is_torch_version("<", "1.12.0"):
                raise ValueError("FSDP requires PyTorch >= 1.12.0")

        if fsdp_plugin is None:  # init from env variables
            fsdp_plugin = FullyShardedDataParallelPlugin() if os.environ.get("USE_FSDP", "false") == "true" else None
        else:
            if not isinstance(fsdp_plugin, FullyShardedDataParallelPlugin):
                raise TypeError("`fsdp_plugin` must be a FullyShardedDataParallelPlugin object.")
            os.environ["USE_FSDP"] = "true"  # use FSDP if plugin is provided

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(handler, KwargsHandler), f"Unsupported kwargs handler passed: {handler}."
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

        kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
        self.state = AcceleratorState(
            mixed_precision=mixed_precision,
            cpu=cpu,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            _from_accelerator=True,
            **kwargs,
        )

        if (
            (mixed_precision != "bf16")
            and getattr(self.state, "downcast_bfloat", False)
            and (self.state.distributedType != DistributedType.TPU)
        ):
            raise ValueError("Can only use `downcast_bf16` when using `mixed_precision='bf16'` and on a TPU")

        if gradient_accumulation_steps > 1:
            if self.state.distributed_type == DistributedType.TPU:
                raise NotImplementedError(
                    "Gradient accumulation on TPU is not supported. Pass in `gradient_accumulation_steps=1`"
                )

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_placement = device_placement
        self.split_batches = split_batches
        self.dispatch_batches = dispatch_batches
        if dispatch_batches is True and is_torch_version("<", "1.8.0"):
            raise ImportError(
                "Using `DataLoaderDispatcher` requires PyTorch 1.8.0 minimum. You have {torch.__version__}."
            )
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        err = "{mode} mixed precision requires {requirement}"
        if self.state.mixed_precision == "fp16":
            self.native_amp = is_torch_version(">=", "1.6")
            if not self.native_amp:
                raise ValueError(err.format(mode="fp16", requirement="PyTorch >= 1.6"))
            if not torch.cuda.is_available() and not parse_flag_from_env("USE_MPS_DEVICE"):
                raise ValueError(err.format(mode="fp16", requirement="a GPU"))
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            if self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler(**kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)
        elif self.state.mixed_precision == "bf16" and self.distributed_type != DistributedType.FSDP:
            self.native_amp = is_bf16_available(True)
            if mixed_precision == "bf16" and not self.native_amp and not is_tpu_available():
                raise ValueError(err.format(mode="bf16", requirement="PyTorch >= 1.10 and a supported device."))

            # Only on the GPU do we care about scaling the gradients
            if torch.cuda.is_available() and self.device.type != "cpu":
                kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)

        # Start of internal step tracking
        self.step = 0
        self.gradient_state = GradientState()

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._custom_objects = []

        # RNG Types
        self.rng_types = rng_types
        if self.rng_types is None:
            self.rng_types = ["torch"] if is_torch_version("<=", "1.5.1") else ["generator"]

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.distributed_type != DistributedType.NO and self.num_processes > 1

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
    def is_main_process(self):
        """True for one process only."""
        return self.process_index == 0

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.local_process_index == 0

    @property
    def use_fp16(self):
        return self.mixed_precision != "no"

    @property
    def mixed_precision(self):
        if self.distributed_type == DistributedType.DEEPSPEED:
            config = self.state.deepspeed_plugin.deepspeed_config
            if config.get("fp16", {}).get("enabled", False):
                mixed_precision = "fp16"
            elif config.get("bf16", {}).get("enabled", False):
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"
        else:
            mixed_precision = self.state.mixed_precision
        return mixed_precision

    def on_main_process(func):
        """
        A decorator that will run the decorated function on the main process only.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.is_main_process or not self.use_distributed:
                return func(self, *args, **kwargs)

        return wrapper

    def on_local_main_process(func):
        """
        A decorator that will run the decorated function on the local main process only.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.is_local_main_process or not self.use_distributed:
                return func(self, *args, **kwargs)

        return wrapper

    def on_process(process_idx):
        """
        A decorator that will run the decorated function on a given process index only.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if self.process_idx == process_idx or not self.use_distributed:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def on_local_process(local_process_idx):
        """
        A decorator that will run the decorated function on a given local process index only.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if self.local_process_idx == local_process_idx or not self.use_distributed:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def _goes_first(self, is_main):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield from self._goes_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield from self._goes_first(self.is_local_main_process)

    @contextmanager
    def no_sync(self, model):
        """
        A context manager to disable gradient synchronizations across DDP processes by calling
        `torch.nn.parallel.DistributedDataParallel.no_sync`.

        If `model` is not in DDP, this context manager does nothing

        Args:
            model (`torch.nn.Module`):
                PyTorch Module that was prepared with `Accelerator.prepare`
        """
        context = contextlib.nullcontext
        if self.use_distributed:
            context = getattr(model, "no_sync", context)

        with context():
            yield

    def _do_sync(self):
        "Sets the right `sync_gradients` context and either resets or increases `self.step`"
        if self.gradient_state.end_of_dataloader:
            self.step = 0
            self.gradient_state._set_sync_gradients(True)
        else:
            self.step += 1
            self.gradient_state._set_sync_gradients((self.step % self.gradient_accumulation_steps) == 0)

    @property
    def sync_gradients(self):
        return self.gradient_state.sync_gradients

    @contextmanager
    def accumulate(self, model):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            model (`torch.nn.Module`):
                PyTorch Module that was prepared with `Accelerator.prepare`
        """
        self._do_sync()
        if self.sync_gradients:
            context = contextlib.nullcontext
        else:
            context = self.no_sync

        with context(model):
            yield

    def print(self, *args, **kwargs):
        """
        Use in replacement of `print()` to only print once per server.
        """
        if self.is_local_main_process:
            print(*args, **kwargs)

    def _prepare_one(self, obj, first_pass=False):
        # First pass of preparation: DataLoader, model, optimizer
        if first_pass:
            if isinstance(obj, torch.utils.data.DataLoader):
                return self.prepare_data_loader(obj)
            elif isinstance(obj, torch.nn.Module):
                return self.prepare_model(obj)
            elif isinstance(obj, torch.optim.Optimizer):
                optimizer = self.prepare_optimizer(obj)
                return optimizer
        # Second pass of preparation: LR scheduler (which need the full list of optimizers)
        elif isinstance(obj, torch.optim.lr_scheduler._LRScheduler):
            scheduler = self.prepare_scheduler(obj)
            return scheduler
        # Return the unprocessed object if previous criteria was not met
        return obj

    def _prepare_fsdp(self, *args):
        result = []
        for obj in args:
            if isinstance(obj, torch.nn.Module):
                model = obj
                break
        optimizers = []

        self._schedulers = []
        self._models = []
        intermediate_result = []
        for obj in args:
            if isinstance(obj, torch.optim.Optimizer):
                if len(obj.param_groups) > 1:
                    logger.warn(
                        "FSDP Warning: When using FSDP, several parameter groups will be conflated into "
                        "a single one due to nested module wrapping and parameter flattening."
                    )
                optimizer = obj.optimizer.__class__(model.parameters(), **obj.optimizer.defaults)
                obj = self.prepare_optimizer(optimizer)
                optimizers.append(obj)
            elif isinstance(obj, torch.nn.Module):
                self._models.append(obj)
            intermediate_result.append(obj)

        for obj in intermediate_result:
            if isinstance(obj, AcceleratedScheduler):
                obj.optimizer = optimizers
                for i, opt in enumerate(self._optimizers):
                    if getattr(obj.scheduler, "optimizer", None) == opt.optimizer:
                        obj.scheduler.optimizer = optimizers[i]
                        obj.optimizers = [optimizers[i]]
                        break
                self._schedulers.append(obj)
            result.append(obj)
        self._optimizers = optimizers
        return tuple(result)

    def prepare(self, *args):
        """
        Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
        order.

        Accepts the following type of objects:

            - `torch.utils.data.DataLoader`: PyTorch Dataloader
            - `torch.nn.Module`: PyTorch Module
            - `torch.optim.Optimizer`: PyTorch Optimizer
        """
        if self.distributed_type == DistributedType.FSDP:
            model_count = 0
            optimizer_present = False
            for obj in args:
                if isinstance(obj, torch.nn.Module):
                    model_count += 1
                if isinstance(obj, torch.optim.Optimizer):
                    optimizer_present = True
            if model_count > 1 and optimizer_present:
                raise ValueError(
                    "For FSDP to work with multiple models (>1), "
                    "prepare must be called for all the models before optimizers are created. "
                    "Then pass the optimizers to the prepare call in the same order as corresponding models."
                )
            elif model_count == 1 and optimizer_present:
                logger.warn(
                    "FSDP Warning: When using FSDP, "
                    "it is efficient and recommended to call prepare for the model before creating the optimizer"
                )

        # On TPUs, putting the model on the XLA device will create new parameters, so the corresponding optimizer will
        # have parameters disconnected from the model (so no training :-( ).
        # If the model and optimizer have parameters on different devices we raise an error.
        if self.distributed_type == DistributedType.TPU:
            model_device, optimizer_device = self._get_devices()
            if model_device is not None and optimizer_device is not None and model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you "
                    "created an optimizer around your model **before** putting on the device. Make sure the line "
                    "model.to(device) is before the optimizer creation in your script or remove it entirely and use "
                    "the flag default value for `devicement_placement` in your `Accelerator` to let it handle that "
                    "part for you."
                )

        # If we're dealing with device placement, this deals with that by...
        tpu_should_fix_optimizer = self.device_placement and self.distributed_type == DistributedType.TPU
        if tpu_should_fix_optimizer:
            # 1. grabbing old model parameters
            old_named_params = self._get_named_parameters(*args)

        if self.distributed_type == DistributedType.DEEPSPEED:
            result = self._prepare_deepspeed(*args)
        else:
            result = tuple(self._prepare_one(obj, first_pass=True) for obj in args)
            result = tuple(self._prepare_one(obj) for obj in result)

        if tpu_should_fix_optimizer:
            # 2. grabbing new model parameters
            new_named_params = self._get_named_parameters(*result)
            # 3. building a map from the first to the second
            mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
            # 4. using that map to update the parameters of the optimizer
            for obj in result:
                if isinstance(obj, torch.optim.Optimizer):
                    obj._switch_parameters(mapping)

        if self.distributed_type == DistributedType.FSDP and model_count == 1 and optimizer_present:
            result = self._prepare_fsdp(*result)

        return result if len(result) > 1 else result[0]

    def prepare_model(self, model):
        self._models.append(model)
        if self.device_placement and self.distributed_type != DistributedType.FSDP:
            model = model.to(self.device)
        if self.distributed_type == DistributedType.MULTI_GPU:
            kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
            )
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
            # don't wrap it again
            if type(model) != FSDP:
                self.state.fsdp_plugin.set_auto_wrap_policy(model)
                fsdp_plugin = self.state.fsdp_plugin
                model = FSDP(
                    model,
                    sharding_strategy=fsdp_plugin.sharding_strategy,
                    cpu_offload=fsdp_plugin.cpu_offload,
                    auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
                    backward_prefetch=fsdp_plugin.backward_prefetch,
                    mixed_precision=fsdp_plugin.mixed_precision_policy,
                    ignored_modules=fsdp_plugin.ignored_modules,
                )
                if not fsdp_plugin.cpu_offload.offload_params:
                    model.to(self.device)
            self._models[-1] = model
        elif self.distributed_type == DistributedType.MULTI_CPU:
            kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        if self.native_amp:
            if self.mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
                model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
            elif self.mixed_precision == "bf16" and self.distributed_type != DistributedType.TPU:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                model.forward = torch.autocast(device_type=device_type, dtype=torch.bfloat16)(model.forward)
            else:
                model.forward = torch.cuda.amp.autocast()(model.forward)
            model.forward = convert_outputs_to_fp32(model.forward)
        if self.distributed_type == DistributedType.TPU and self.state.fork_launched:
            model = xmp.MpModelWrapper(model).to(self.device)
        return model

    def _prepare_deepspeed(self, *args):

        deepspeed_plugin = self.state.deepspeed_plugin

        result = [
            self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
            for obj in args
        ]

        batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
        if self.split_batches:
            batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]
        if len(batch_sizes) == 0:
            raise ValueError(
                "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using DeepSpeed."
            )

        batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
        if len(batch_sizes) > 1:
            logger.info(
                "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
            )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
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
            elif (isinstance(obj, (torch.optim.lr_scheduler._LRScheduler, DummyScheduler))) or (
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
            elif "scheduler" not in deepspeed_plugin.deepspeed_config and isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot create a `DummyScheduler` without specifying a scheduler in the config file."
                )

        if optimizer is not None and scheduler is not None:
            if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
                    "`accelerate.utils.DummyOptim`."
                )

        if model is not None:
            if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
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
            if isinstance(scheduler, (DummyScheduler)):
                config_kwargs.update(
                    {
                        "scheduler.params.warmup_min_lr": 0,
                        "scheduler.params.warmup_max_lr": scheduler.optimizer.lr,
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
                else:
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
                elif (isinstance(result[i], (torch.optim.lr_scheduler._LRScheduler, DummyScheduler))) or (
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

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement if self.distributed_type != DistributedType.TPU else False,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
        )

    def prepare_optimizer(self, optimizer):
        optimizer = AcceleratedOptimizer(optimizer, device_placement=self.device_placement, scaler=self.scaler)
        self._optimizers.append(optimizer)
        return optimizer

    def prepare_scheduler(self, scheduler):
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
        Use `accelerator.backward(loss)` in lieu of `loss.backward()`.
        """
        loss /= self.gradient_accumulation_steps
        if self.distributed_type == DistributedType.DEEPSPEED:
            self.deepspeed_engine_wrapped.backward(loss, **kwargs)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def unscale_gradients(self, optimizer=None):
        """
        Unscale the gradients in mixed precision training with AMP. This is a noop in all other settings.

        Args:
            optimizer (`torch.optim.Optimizer` or `List[torch.optim.Optimizer]`, *optional*):
                The optimizer(s) for which to unscale gradients. If not set, will unscale gradients on all optimizers
                that were passed to [`~Accelerator.prepare`].
        """
        if self.use_fp16 and self.native_amp:
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
        """
        if self.distributed_type == DistributedType.FSDP:
            self.unscale_gradients()
            parameters = [p for p in parameters]
            for model in self._models:
                if parameters == [p for p in model.parameters()]:
                    model.clip_grad_norm_(max_norm, norm_type)
                    return
        elif self.distributed_type == DistributedType.DEEPSPEED:
            # `accelerator.backward(loss)` is doing that automatically. Therefore, it's implementation is not needed
            return
        self.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of `torch.nn.utils.clip_grad_value_`.
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
        """
        return gather(tensor)

    def gather_for_metrics(self, tensor):
        """
        Gathers `tensor` and potentially drops duplicates in the last batch if on a distributed system. Should be used
        for gathering the inputs and targets for metric calculation.

        Args:
            tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
                The tensors for calculating metrics across all processes.
        """
        tensor = self.gather(tensor)
        if self.use_distributed:
            if self.gradient_state.remainder == -1:
                logger.info(
                    "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                )
                return tensor
            try:
                # Then see if we're on the last batch of our eval dataloader
                if self.gradient_state.end_of_dataloader:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder]

                    return recursively_apply(_adjust_samples, tensor)
                else:
                    # Not at the end of the dataloader, no need to adjust the tensors
                    return tensor
            except:
                # Dataset had no length or raised an error
                return tensor
        return tensor

    def reduce(self, tensor, reduction="sum"):
        """
        Reduce the values in *tensor* across all processes based on *reduction*.

        Note:
            All processes get the reduced value.

        Args:
            tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
                The tensors to reduce across all processes.
            reduction (`str`, *optional*, defaults to "sum"):
                A reduction type, can be one of 'sum', 'mean', or 'none'. If 'none', will not perform any operation.

        Returns:
            `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`: The reduced tensor(s).
        """
        return reduce(tensor, reduction)

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
        """
        return pad_across_processes(tensor, dim=dim, pad_index=pad_index, pad_first=pad_first)

    def unwrap_model(self, model):
        """
        Unwraps the `model` from the additional layer possible added by [`~Accelerator.prepare`]. Useful before saving
        the model.

        Args:
            model (`torch.nn.Module`):
                The model to unwrap.
        """
        return extract_model_from_parallel(model)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.
        """
        wait_for_everyone()

    def init_trackers(self, project_name: str, config: Optional[dict] = None, init_kwargs: Optional[dict] = {}):
        """
        Initializes a run for all trackers stored in `self.log_with`, potentially with starting configurations

        Args:
            project_name (`str`):
                The name of the project. All trackers will save their data based on this
            config (`dict`, *optional*):
                Optional starting configuration to be logged.
            init_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `__init__` function. Should be
                formatted like this:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```
        """
        self.trackers = []
        for tracker in self.log_with:
            if issubclass(type(tracker), GeneralTracker):
                # Custom trackers are already initialized
                self.trackers.append(tracker)
            else:
                tracker_init = LOGGER_TYPE_TO_CLASS[str(tracker)]
                if getattr(tracker_init, "requires_logging_directory"):
                    # We can skip this check since it was done in `__init__`
                    self.trackers.append(
                        tracker_init(project_name, self.logging_dir, **init_kwargs.get(str(tracker), {}))
                    )
                else:
                    self.trackers.append(tracker_init(project_name, **init_kwargs.get(str(tracker), {})))
        if config is not None:
            for tracker in self.trackers:
                tracker.store_init_configuration(config)

    @on_main_process
    def get_tracker(self, name: str):
        """
        Returns a `tracker` from `self.trackers` based on `name` on the main process only.

        Args:
            name (`str`):
                The name of a tracker, corresponding to the `.name` property.
        """
        for tracker in self.trackers:
            if tracker.name == name:
                return tracker.tracker
        raise ValueError(f"{name} is not an available tracker stored inside the `Accelerator`.")

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, log_kwargs: Optional[dict] = {}):
        """
        Logs `values` to all stored trackers in `self.trackers` on the main process only.

        Args:
            values (`dict`):
                Values should be a dictionary-like object containing only types `int`, `float`, or `str`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            log_kwargs (`dict`, *optional*):
                A nested dictionary of kwargs to be passed to a specific tracker's `log` function. Should be formatted
                like this:
                ```python
                {"wandb": {"tags": ["tag_a", "tag_b"]}}
                ```
        """
        for tracker in self.trackers:
            tracker.log(values, step=step, **log_kwargs.get(tracker.name, {}))

    @on_main_process
    def end_training(self):
        """
        Runs any special end training behaviors, such as stopping trackers on the main process only.
        """
        for tracker in self.trackers:
            tracker.finish()

    def save(self, obj, f):
        """
        Save the object passed to disk once per machine. Use in place of `torch.save`.

        Args:
            obj: The object to save.
            f (`str` or `os.PathLike`):
                Where to save the content of `obj`.
        """
        save(obj, f)

    def save_state(self, output_dir: str):
        """
        Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects.

        Args:
            output_dir (`str` or `os.PathLike`):
                The name of the folder to save all relevant weights and states.
        """
        # Check if folder exists
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Saving FSDP model")
                self.state.fsdp_plugin.save_model(self, model, output_dir, i)
                logger.info(f"FSDP Model saved to output dir {output_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Saving DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.save_checkpoint(output_dir, ckpt_id)
                logger.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, ckpt_id)}")
            else:
                weights.append(self.get_state_dict(model, unwrap=False))

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for opt in self._optimizers:
                logger.info("Saving FSDP Optimizer")
                self.state.fsdp_plugin.save_optimizer(self, opt, self._models[i], output_dir, i)
                logger.info(f"FSDP Optimizer saved to output dir {output_dir}")
        elif self.distributed_type != DistributedType.DEEPSPEED:
            optimizers = self._optimizers

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        else:
            schedulers = self._schedulers

        save_location = save_accelerator_state(
            output_dir, weights, optimizers, schedulers, self.state.process_index, self.scaler
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i)
        return save_location

    def load_state(self, input_dir: str):
        """
        Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.

        Args:
            input_dir (`str` or `os.PathLike`):
                The name of the folder all relevant weights and states were saved in.
        """
        # Check if folder exists
        input_dir = os.path.expanduser(input_dir)
        if not os.path.isdir(input_dir):
            raise ValueError(f"Tried to find {input_dir} but folder does not exist")
        logger.info(f"Loading states from {input_dir}")

        # Load the models taking care of FSDP and DeepSpeed nuances
        models = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Loading FSDP model")
                self.state.fsdp_plugin.load_model(self, model, input_dir, i)
                logger.info(f"FSDP Model loaded from input dir {input_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Loading DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.load_checkpoint(input_dir, ckpt_id)
                logger.info(f"DeepSpeed Model and Optimizer loaded from input dir {os.path.join(input_dir, ckpt_id)}")
            else:
                models.append(model)

        # Load the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(self._optimizers):
                logger.info("Loading FSDP Optimizer")
                self.state.fsdp_plugin.load_optimizer(self, opt, self._models[i], input_dir, i)
                logger.info(f"FSDP Optimizer loaded from input dir {input_dir}")
        elif self.distributed_type != DistributedType.DEEPSPEED:
            optimizers = self._optimizers

        # Load the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        else:
            schedulers = self._schedulers

        load_accelerator_state(input_dir, models, optimizers, schedulers, self.state.process_index, self.scaler)
        custom_checkpoints = [f for f in os.listdir(input_dir) if "custom_checkpoint" in f]
        if len(custom_checkpoints) != len(self._custom_objects):
            err = "Warning! Number of found checkpoints does not match the number of registered objects:"
            err += f"\n\tFound checkpoints: {len(custom_checkpoints)}"
            err += f"\n\tRegistered objects: {len(self._custom_objects)}\nSkipping."
            logger.warn(err)
        else:
            logger.info(f"Loading in {len(custom_checkpoints)} custom states")
            for index, obj in enumerate(self._custom_objects):
                load_custom_state(obj, input_dir, index)

    def free_memory(self):
        """
        Will release all references to the internal objects stored and call the garbage collector. You should call this
        method between two trainings with different models/optimizers.
        """
        self._schedulers = []
        self._optimizers = []
        self._models = []
        self.deepspeed_engine_wrapped = None
        gc.collect()
        torch.cuda.empty_cache()

    def clear(self):
        """
        Alias for [`Accelerate.free_memory`], releases all references to the internal objects stored and call the
        garbage collector. You should call this method between two trainings with different models/optimizers.
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
        is_zero_3 = False
        if self.distributed_type == DistributedType.DEEPSPEED:
            is_zero_3 = self.deepspeed_config["zero_optimization"]["stage"] == 3

        if is_zero_3:
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
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()

        if state_dict is not None:
            for k in state_dict:
                if state_dict[k].dtype == torch.float16:
                    state_dict[k] = state_dict[k].float()

        return state_dict

    def register_for_checkpointing(self, *objects):
        """
        Makes note of `objects` and will save or load them in during `save_state` or `load_state`.

        These should be utilized when the state is being loaded or saved in the same script. It is not designed to be
        used in different scripts

        <Tip>

        Every `object` must have a `load_state_dict` and `state_dict` function to be stored.

        </Tip>
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
    def autocast(self):
        """
        Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
        different will happen otherwise.
        """
        if self.native_amp:
            if self.mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
                autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
            elif self.mixed_precision == "bf16" and is_bf16_available():
                if self.distributed_type in [DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU]:
                    device_type = "cpu" if not torch.cuda.is_available() else "cuda"
                    autocast_context = torch.autocast(dtype=torch.bfloat16, device_type=device_type)
            else:
                autocast_context = torch.cuda.amp.autocast()

            autocast_context.__enter__()
            yield
            autocast_context.__exit__(*sys.exc_info())
        else:
            yield

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
