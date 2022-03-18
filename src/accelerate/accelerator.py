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

import gc
import os
import sys
import warnings
from contextlib import contextmanager
from typing import List, Optional, Union

import torch

from packaging import version

from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import prepare_data_loader
from .kwargs_handlers import DistributedDataParallelKwargs, GradScalerKwargs, InitProcessGroupKwargs, KwargsHandler
from .optimizer import AcceleratedOptimizer
from .state import AcceleratorState, DistributedType, is_deepspeed_available
from .utils import (
    DeepSpeedPlugin,
    RNGType,
    convert_outputs_to_fp32,
    extract_model_from_parallel,
    gather,
    get_pretty_name,
    pad_across_processes,
    save,
    wait_for_everyone,
)


if is_deepspeed_available():
    import deepspeed

    from .deepspeed_utils import DeepSpeedEngineWrapper, DeepSpeedOptimizerWrapper

import logging


logger = logging.getLogger(__name__)


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
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
            the execution on one process only.
        deepspeed_plugin (`DeepSpeedPlugin`, *optional*):
            Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
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
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        kwargs_handlers (`List[KwargHandler]`, *optional*)
            A list of `KwargHandler` to customize how the objects related to distributed training or mixed precision
            are created. See [kwargs](kwargs) for more information.

    Attributes

        - **device** (`torch.device`) -- The device to use.
        - **state** ([`~state.AcceleratorState`]) -- The distributed setup state.
    """

    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = False,
        fp16: bool = None,
        mixed_precision: str = None,
        cpu: bool = False,
        deepspeed_plugin: DeepSpeedPlugin = None,
        rng_types: Optional[List[Union[str, RNGType]]] = None,
        dispatch_batches: Optional[bool] = None,
        kwargs_handlers: Optional[List[KwargsHandler]] = None,
    ):

        if mixed_precision is not None:
            mixed_precision = mixed_precision.lower()
            if mixed_precision not in ["no", "fp16", "bf16"]:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between 'no', 'fp16' and 'bf16'."
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
            _from_accelerator=True,
            **kwargs,
        )

        self.device_placement = device_placement
        self.split_batches = split_batches
        self.dispatch_batches = dispatch_batches
        if dispatch_batches is True and version.parse(torch.__version__) < version.parse("1.8.0"):
            raise ImportError(
                "Using `DataLoaderDispatcher` requires PyTorch 1.8.0 minimum. You have {torch.__version__}."
            )

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if self.state.mixed_precision == "fp16":
            self.native_amp = version.parse(torch.__version__) >= version.parse("1.6")
            if version.parse(torch.__version__) < version.parse("1.6"):
                raise ValueError("fp16 mixed precision requires PyTorch >= 1.6")

            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            self.scaler = torch.cuda.amp.GradScaler(**kwargs)
        elif self.state.mixed_precision == "bf16":
            self.native_amp = version.parse(torch.__version__) >= version.parse("1.10")
            if mixed_precision == "bf16" and version.parse(torch.__version__) < version.parse("1.10"):
                raise ValueError("bf16 mixed precision requires PyTorch >= 1.10")

            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            self.scaler = torch.cuda.amp.GradScaler(**kwargs)

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._custom_objects = []

        # RNG Types
        self.rng_types = rng_types
        if self.rng_types is None:
            self.rng_types = ["torch"] if version.parse(torch.__version__) <= version.parse("1.5.1") else ["generator"]

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
            if self.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
                mixed_precision = "fp16"
            elif self.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"
        else:
            mixed_precision = self.state.mixed_precision
        return mixed_precision

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield from self._goes_first(self.is_local_main_process)

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield from self._goes_first(self.is_main_process)

    def _goes_first(self, is_main):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    def print(self, *args, **kwargs):
        """
        Use in replacement of `print()` to only print once per server.
        """
        if self.is_local_main_process:
            print(*args, **kwargs)

    def _prepare_one(self, obj):
        if isinstance(obj, torch.utils.data.DataLoader):
            return self.prepare_data_loader(obj)
        elif isinstance(obj, torch.nn.Module):
            self._models.append(obj)
            return self.prepare_model(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = self.prepare_optimizer(obj)
            self._optimizers.append(optimizer)
            return optimizer
        else:
            return obj

    def prepare(self, *args):
        """
        Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
        order.

        Accepts the following type of objects:

            - `torch.utils.data.DataLoader`: PyTorch Dataloader
            - `torch.nn.Module`: PyTorch Module
            - `torch.optim.Optimizer`: PyTorch Optimizer
        """
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
            result = tuple(self._prepare_one(obj) for obj in args)

        if tpu_should_fix_optimizer:
            # 2. grabbing new model parameters
            new_named_params = self._get_named_parameters(*result)
            # 3. building a map from the first to the second
            mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
            # 4. using that map to update the parameters of the optimizer
            for obj in result:
                if isinstance(obj, torch.optim.Optimizer):
                    obj._switch_parameters(mapping)

        return result if len(result) > 1 else result[0]

    def prepare_model(self, model):
        if self.device_placement:
            model = model.to(self.device)
        if self.distributed_type == DistributedType.MULTI_GPU:
            kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
            )
        elif self.distributed_type == DistributedType.MULTI_CPU:
            kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        if self.native_amp:
            if self.mixed_precision == "fp16" and version.parse(torch.__version__) >= version.parse("1.10"):
                model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
            elif self.mixed_precision == "bf16":
                model.forward = torch.cuda.amp.autocast(dtype=torch.bfloat16)(model.forward)
            else:
                model.forward = torch.cuda.amp.autocast()(model.forward)
            model.forward = convert_outputs_to_fp32(model.forward)
        return model

    def _prepare_deepspeed(self, *args):

        deepspeed_plugin = self.state.deepspeed_plugin
        self.deepspeed_config = deepspeed_plugin.deepspeed_config

        batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
        if len(batch_sizes) == 0:
            raise ValueError(
                "You must specify a training or evaluation dataloader in `accelerate.prepare()` when using DeepSpeed."
            )

        batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
        if len(batch_sizes) > 1:
            logger.info(
                f"Since you passed both train and evaluation dataloader, `is_train_batch_min` (here \
                {deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
            )

        self.deepspeed_config["train_batch_size"] = (
            batch_size_per_device * deepspeed_plugin.gradient_accumulation_steps * self.num_processes
        )

        result = [self._prepare_one(obj) if isinstance(obj, torch.utils.data.DataLoader) else obj for obj in args]

        model = None
        optimizer = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, dict)):
                optimizer = obj

        if deepspeed_plugin.auto_opt_mapping:
            is_adam = isinstance(optimizer, torch.optim.Adam)
            is_adamw = isinstance(optimizer, torch.optim.AdamW)
            if (is_adam or is_adamw) and deepspeed_plugin.offload_optimizer_device == "cpu":
                defaults = optimizer.defaults
                params = []
                for group in optimizer.param_groups:
                    params.extend(group["params"])

                optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                    params,
                    lr=defaults["lr"],
                    bias_correction=True,
                    betas=defaults["betas"],
                    eps=defaults["eps"],
                    weight_decay=defaults["weight_decay"],
                    amsgrad=defaults["amsgrad"],
                    adamw_mode=is_adamw,
                )

        # useful when only eval_dataloader is given into `accelerator.prepare()`
        if model is not None:
            engine = DeepSpeedEngineWrapper(
                args=None,
                model=model,
                optimizer=optimizer,
                config_params=self.deepspeed_config,
                dist_init_required=False,
            )
            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], torch.optim.Optimizer):
                    result[i] = DeepSpeedOptimizerWrapper(engine.optimizer, engine)
            self.deepspeed_engine = engine  # pointing for deepspeed_engine.backward()
            self._models.append(engine)
            self._optimizers.append(engine.optimizer)
            assert (
                len(self._models) == 1
            ), "You can't use same `Accelerator()` instance with 2 models when using DeepSpeed"

        if self.distributed_type == DistributedType.DEEPSPEED:
            assert hasattr(
                self, "deepspeed_engine"
            ), "You need to pass the model along the optimizer when using Deepspeed."

        return tuple(result)

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
        )

    def prepare_optimizer(self, optimizer):
        return AcceleratedOptimizer(optimizer, device_placement=self.device_placement, scaler=self.scaler)

    def backward(self, loss, **kwargs):
        """
        Use `accelerator.backward(loss)` in lieu of `loss.backward()`.
        """
        if self.distributed_type == DistributedType.DEEPSPEED:
            self.deepspeed_engine.backward(loss, **kwargs)
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
        self.unscale_gradients()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of `torch.nn.utils.clip_grad_value_`.
        """
        self.unscale_gradients()
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    def gather(self, tensor):
        """
        Gather the values in *tensor* accross all processes and concatenate them on the first dimension. Useful to
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
        weights = [self.get_state_dict(m) for m in self._models]
        save_location = save_accelerator_state(
            output_dir, weights, self._optimizers, self.state.process_index, self.scaler
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
        load_accelerator_state(input_dir, self._models, self._optimizers, self.state.process_index, self.scaler)
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
        self._optimizers = []
        self._models = []
        self.deepspeed_engine = None
        gc.collect()
        torch.cuda.empty_cache()

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

    def get_state_dict(self, model):
        is_zero_3 = False
        if is_deepspeed_available():
            if isinstance(model, DeepSpeedEngineWrapper) and self.distributed_type == DistributedType.DEEPSPEED:
                is_zero_3 = self.state.deepspeed_plugin.zero_stage == 3

        if is_zero_3:
            state_dict = model._zero3_consolidated_fp16_state_dict()
        else:
            model = self.unwrap_model(model)
            state_dict = model.state_dict()

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
            if self.mixed_precision == "fp16" and version.parse(torch.__version__) >= version.parse("1.10"):
                autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)
            elif self.mixed_precision == "bf16":
                autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16)
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
            if optimizer.is_overflow:
                return True
        return False
