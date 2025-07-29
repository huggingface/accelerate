# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import copy
import functools
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch

from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, OPTIMIZER_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import get_module_class_from_name
from .modeling import get_non_persistent_buffers, is_peft_model
from .other import get_module_children_bottom_up, is_compiled_module, save
from .versions import is_torch_version


logger = get_logger(__name__)


def enable_fsdp_ram_efficient_loading():
    """
    Enables RAM efficient loading of Hugging Face models for FSDP in the environment.
    """
    # Sets values for `transformers.modeling_utils.is_fsdp_enabled`
    if "ACCELERATE_USE_FSDP" not in os.environ:
        os.environ["ACCELERATE_USE_FSDP"] = "True"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"


def disable_fsdp_ram_efficient_loading():
    """
    Disables RAM efficient loading of Hugging Face models for FSDP in the environment.
    """
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "False"


def _get_model_state_dict(model, adapter_only=False, sd_options=None):
    if adapter_only and is_peft_model(model):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(model, adapter_name=model.active_adapter)

    # Invariant: `sd_options` is not None only for FSDP2
    if sd_options is not None:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        return get_model_state_dict(model, options=sd_options)
    else:
        return model.state_dict()


def _set_model_state_dict(model, state_dict, adapter_only=False, sd_options=None):
    if adapter_only and is_peft_model(model):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(model, state_dict, adapter_name=model.active_adapter)

    # Invariant: `sd_options` is not None only for FSDP2
    if sd_options is not None:
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        return set_model_state_dict(model, state_dict, options=sd_options)
    else:
        return model.load_state_dict(state_dict)


def _prepare_sd_options(fsdp_plugin):
    sd_options = None

    # we use this only for FSDP2, as it requires torch >= 2.6.0 and this api requires torch >= 2.2.0
    if fsdp_plugin.fsdp_version == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        sd_options = StateDictOptions(
            full_state_dict=fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT,
            cpu_offload=getattr(fsdp_plugin.state_dict_config, "offload_to_cpu", False),
            broadcast_from_rank0=getattr(fsdp_plugin.state_dict_config, "rank0_only", False),
        )

    return sd_options


def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    os.makedirs(output_dir, exist_ok=True)
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        state_dict = _get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                logger.info(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
        # Invariant: `LOCAL_STATE_DICT` is never possible with `FSDP2`
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            output_model_file = os.path.join(output_dir, weights_name)
            logger.info(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{model_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving model to {ckpt_dir}")
            state_dict = {"model": state_dict}

            dist_cp.save(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Model saved to {ckpt_dir}")


def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)
    with ctx:
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) is not FSDP and accelerator.process_index != 0 and not accelerator.is_fsdp2:
                if not fsdp_plugin.sync_module_states and fsdp_plugin.fsdp_version == 1:
                    raise ValueError(
                        "Set the `sync_module_states` flag to `True` so that model states are synced across processes when "
                        "initializing FSDP object"
                    )
                return
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            # we want an empty state dict for FSDP2 as we use `broadcast_from_rank0`
            load_model = not accelerator.is_fsdp2 or accelerator.is_main_process
            if load_model:
                state_dict = torch.load(input_model_file, weights_only=True)
            else:
                state_dict = {}
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file, weights_only=True)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = (
                os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}")
                if f"{FSDP_MODEL_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading model from {ckpt_dir}")
            state_dict = {"model": _get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)}
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            state_dict = state_dict["model"]
            logger.info(f"Model loaded from {ckpt_dir}")

        load_result = _set_model_state_dict(model, state_dict, adapter_only=adapter_only, sd_options=sd_options)
    return load_result


def save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    os.makedirs(output_dir, exist_ok=True)

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )

    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        if fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
        else:
            optim_state = FSDP.optim_state_dict(model, optimizer)

        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if accelerator.process_index == 0:
                optim_state_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                output_optimizer_file = os.path.join(output_dir, optim_state_name)
                logger.info(f"Saving Optimizer state to {output_optimizer_file}")
                torch.save(optim_state, output_optimizer_file)
                logger.info(f"Optimizer state saved in {output_optimizer_file}")
        else:
            ckpt_dir = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving Optimizer state to {ckpt_dir}")
            dist_cp.save(
                state_dict={"optimizer": optim_state},
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Optimizer state saved in {ckpt_dir}")


def load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir, optimizer_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)
    with ctx:
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            optim_state = None
            if accelerator.process_index == 0 or not fsdp_plugin.optim_state_dict_config.rank0_only:
                optimizer_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                input_optimizer_file = os.path.join(input_dir, optimizer_name)
                logger.info(f"Loading Optimizer state from {input_optimizer_file}")
                optim_state = torch.load(input_optimizer_file, weights_only=True)
                logger.info(f"Optimizer state loaded from {input_optimizer_file}")
        else:
            ckpt_dir = (
                os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
                if f"{OPTIMIZER_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading Optimizer from {ckpt_dir}")
            optim_state = {"optimizer": optimizer.state_dict()}
            dist_cp.load(
                optim_state,
                checkpoint_id=ckpt_dir,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
            )
            optim_state = optim_state["optimizer"]
            logger.info(f"Optimizer loaded from {ckpt_dir}")

        if fsdp_plugin.fsdp_version == 1:
            flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=optim_state)
            optimizer.load_state_dict(flattened_osd)
        else:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

            set_optimizer_state_dict(model, optimizer, optim_state, options=sd_options)


def _distributed_checkpoint_to_merged_weights(checkpoint_dir: str, save_path: str, safe_serialization: bool = True):
    """
    Passthrough to `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`

    Will save under `save_path` as either `model.safetensors` or `pytorch_model.bin`.
    """
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.format_utils as dist_cp_format_utils

    state_dict = {}
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    dist_cp_format_utils._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=dist_cp_format_utils._EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    save_path = save_path / SAFE_WEIGHTS_NAME if safe_serialization else save_path / WEIGHTS_NAME

    # To handle if state is a dict like {model: {...}}
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]
    save(state_dict, save_path, safe_serialization=safe_serialization)
    return save_path


def merge_fsdp_weights(
    checkpoint_dir: str, output_path: str, safe_serialization: bool = True, remove_checkpoint_dir: bool = False
):
    """
    Merge the weights from sharded FSDP model checkpoints into a single combined checkpoint. Should be used if
    `SHARDED_STATE_DICT` was used for the model. Weights will be saved to `{output_path}/model.safetensors` if
    `safe_serialization` else `pytorch_model.bin`.

    Note: this is a CPU-bound process.

    Args:
        checkpoint_dir (`str`):
            The directory containing the FSDP checkpoints (can be either the model or optimizer).
        output_path (`str`):
            The path to save the merged checkpoint.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the merged weights with safetensors (recommended).
        remove_checkpoint_dir (`bool`, *optional*, defaults to `False`):
            Whether to remove the checkpoint directory after merging.
    """
    checkpoint_dir = Path(checkpoint_dir)
    from accelerate.state import PartialState

    if not is_torch_version(">=", "2.3.0"):
        raise ValueError("`merge_fsdp_weights` requires PyTorch >= 2.3.0`")

    # Verify that the checkpoint directory exists
    if not checkpoint_dir.exists():
        model_path_exists = (checkpoint_dir / "pytorch_model_fsdp_0").exists()
        optimizer_path_exists = (checkpoint_dir / "optimizer_0").exists()
        err = f"Tried to load from {checkpoint_dir} but couldn't find a valid metadata file."
        if model_path_exists and optimizer_path_exists:
            err += " However, potential model and optimizer checkpoint directories exist."
            err += f"Please pass in either {checkpoint_dir}/pytorch_model_fsdp_0 or {checkpoint_dir}/optimizer_0"
            err += "instead."
        elif model_path_exists:
            err += " However, a potential model checkpoint directory exists."
            err += f"Please try passing in {checkpoint_dir}/pytorch_model_fsdp_0 instead."
        elif optimizer_path_exists:
            err += " However, a potential optimizer checkpoint directory exists."
            err += f"Please try passing in {checkpoint_dir}/optimizer_0 instead."
        raise ValueError(err)

    # To setup `save` to work
    state = PartialState()
    if state.is_main_process:
        logger.info(f"Merging FSDP weights from {checkpoint_dir}")
        save_path = _distributed_checkpoint_to_merged_weights(checkpoint_dir, output_path, safe_serialization)
        logger.info(f"Successfully merged FSDP weights and saved to {save_path}")
        if remove_checkpoint_dir:
            logger.info(f"Removing old checkpoint directory {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
    state.wait_for_everyone()


def ensure_weights_retied(param_init_fn, model: torch.nn.Module, device: torch.device):
    _tied_names = getattr(model, "_tied_weights_keys", None)
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):
        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name)
                else:
                    setattr(module, param_name, param)  # tie

        return module

    return param_init_fn_tied_param


def fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    if accelerator.is_main_process:
        for (param_name, full_param), sharded_param in zip(full_sd.items(), meta_sharded_sd.values()):
            device_mesh = sharded_param.device_mesh
            full_param = full_param.detach().to(device_mesh.device_type)
            dist.broadcast(full_param, src=0, group=device_mesh.get_group())
            sharded_tensor = distribute_tensor(full_param, device_mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            device_mesh = sharded_param.device_mesh
            full_tensor = torch.empty(sharded_param.size(), device=device_mesh.device_type, dtype=sharded_param.dtype)
            dist.broadcast(full_tensor, src=0, group=device_mesh.get_group())
            sharded_tensor = distribute_tensor(full_tensor, device_mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    model.load_state_dict(sharded_sd, assign=True)
    return model


def fsdp2_switch_optimizer_parameters(optimizer: torch.optim.Optimizer, mapping: dict):
    """
    Switches the parameters of the optimizer to new ones (sharded parameters in usual case). This function modifies the
    optimizer in-place.

    Args:
        optimizer (`torch.optim.Optimizer`): Optimizer instance which contains the original model parameters
        mapping (`dict`): Mapping from the original parameter (specified by `data_ptr`) to the sharded parameter

    Raises:
        KeyError:
            If a parameter in the optimizer couldn't be switched to its sharded version. This should never happen and
            indicates a bug. If we kept the original params instead of raising, the training wouldn't be numerically
            correct and weights wouldn't get updated.
    """
    from torch.distributed.tensor import DTensor

    accessor_mapping = {}

    accessor_mapping[DTensor] = "_local_tensor"
    try:
        for param_group in optimizer.param_groups:
            param_group["params"] = [mapping[p.data_ptr] for p in param_group["params"]]
    except KeyError:
        # This shouldn't ever happen, but we want to fail here else training wouldn't be numerically correct
        # This basically means that we're missing a mapping from the original parameter to the sharded parameter
        raise KeyError(
            "A parameter in the optimizer couldn't be switched to its sharded version. This breaks the training. Please raise an issue on GitHub."
        )


def fsdp2_apply_ac(accelerator, model: torch.nn.Module):
    """
    Applies the activation checkpointing to the model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to apply the activation checkpointing to

    Returns:
        `torch.nn.Module`: The model with the activation checkpointing applied
    """

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(accelerator.state.fsdp_plugin, model)

    for layer_name, layer in get_module_children_bottom_up(model, return_fqns=True)[:-1]:
        if len(layer_name.split(".")) > 1:
            parent_name, child_name = layer_name.rsplit(".", 1)
        else:
            parent_name = None
            child_name = layer_name

        parent_module = model.get_submodule(parent_name) if parent_name else model
        if auto_wrap_policy_func(parent_module):
            layer = checkpoint_wrapper(layer, preserve_rng_state=False)
            parent_module.register_module(child_name, layer)

    return model


def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare

    Returns:
        `torch.nn.Module`: Prepared model
    """
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    is_type_fsdp = isinstance(model, FSDPModule) or (
        is_compiled_module(model) and isinstance(model._orig_mod, FSDPModule)
    )
    if is_type_fsdp:
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin

    fsdp2_plugin.set_auto_wrap_policy(model)

    original_sd = model.state_dict()
    mesh = getattr(accelerator, "torch_device_mesh", None)

    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
        "mesh": mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
    }

    model_has_params4bit = False
    for name, param in model.named_parameters():
        # this is a temporary fix whereby loading models with bnb params cannot be moved from
        # GPU to a meta device due with FSDP2 because torch operations don't return the original class type
        # bypassing the move to meta will still cause the VRAM spike, but at least it still will load
        if param.__class__.__name__ == "Params4bit":
            model_has_params4bit = True
            break

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # Context: `fully_shard` moves the model to GPU if it was on CPU, however it can also be on `meta` and then it stays there even after `fully_shard`
        # For this reason, we need to move the model to `meta` device, as then sharding happens on `meta` device
        # If we kept the model on CPU (`cpu_ram_efficient_loading` has model be on CPU on all ranks, though non-main ranks only have `torch.emtpy`), `fully_shard` would move it to GPU
        # Afterwards, when we call `fsdp2_load_full_state_dict`, us creating the state_dict would result into briefly having two copies of model state_dict on the GPU -> VRAM spike

        # We need to keep the original non-persistent buffers, as those MAY not be in the state_dict, resulting in them staying on meta device
        # Also, these buffers aren't getting sharded by default
        # We get the FQNs of all non-persistent buffers, to re-register them after
        non_persistent_buffer_fqns = get_non_persistent_buffers(model, recurse=True, fqns=True)
        original_non_persistent_buffers = copy.deepcopy(
            {k: v for k, v in model.named_buffers() if k in non_persistent_buffer_fqns}
        )
        # We move the model to meta device, as then sharding happens on meta device
        model = model.to(torch.device("meta"))
        # We need to re-tie the weights, not exactly sure why, but if we don't do this, reference to `lm_head/embed_tokens` stay hanging -> more VRAM usage
        # We assume `transformers` models have a `tie_weights` method if they support it
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    if auto_wrap_policy_func is not None:
        # We skip the model itself, as that one is always wrapped
        for module in get_module_children_bottom_up(model)[:-1]:
            if auto_wrap_policy_func(module) and not isinstance(module, FSDPModule):
                fully_shard(module, **fsdp2_kwargs)

    if not isinstance(model, FSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        # If `cpu_ram_efficient_loading` is enabled, only rank 0 loads the weights
        # Other ranks have an empty model on `meta` device, so we need to distribute the weights properly
        fsdp2_load_full_state_dict(accelerator, model, original_sd)

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # We re-register the buffers, as they may not be in the state_dict
        for fqn, buffer_tensor in original_non_persistent_buffers.items():
            buffer_tensor = buffer_tensor.to(accelerator.device)

            if "." in fqn:
                parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
                parent_module = model.get_submodule(parent_fqn)
            else:
                local_buffer_name = fqn
                parent_module = model

            parent_module.register_buffer(local_buffer_name, buffer_tensor, persistent=False)

        # We need to tie the weights again, as call to `load_full_state_dict` breaks the tie
        # Needs to be called both here and above
        # removing this call makes the have slightly different loss
        # removing the call above leads to extra memory usage as explained in the comment above
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # There is no `dtype` attribution for nn.Module
    # Set it to None if it doesn't exist and do the upcast always
    model_dtype = getattr(model, "dtype", None)
    if accelerator.mixed_precision != "no" and (model_dtype is None or model_dtype != torch.float32):
        # We upcast the model according to `deepspeed`'s implementation
        # More info about this can be found in `accelerator.py:prepare_model`s FSDP1 section
        model = model.to(torch.float32)
        if accelerator.is_main_process:
            # TODO(siro1): Add a warning for each parameter that was upcasted
            warnings.warn(
                "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no') may affect the precision of model checkpoints."
            )
    return model


def fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model: torch.nn.Module) -> Callable[[torch.nn.Module], bool]:
    """Prepares the auto wrap policy based on its type, done to mimic the behaviour of FSDP1 auto wrap policy.

    Args:
        fsdp2_plugin (`FullyShardedDataParallelPlugin`):
            Instance of `FullyShardedDataParallelPlugin` containing the configuration options
        auto_wrap_policy_type (`str`):
            Either `transformer` or `size`
        model (`torch.nn.Module`):
            The model to wrap

    Returns:
        `Callable[[torch.nn.Module], bool]`:
            The auto wrap policy function to be applied to the model
    """
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

    fn = fsdp2_plugin.auto_wrap_policy

    if isinstance(fn, functools.partial):
        fn = fn.func

    if fn is transformer_auto_wrap_policy:
        no_split_modules = getattr(model, "_no_split_modules", None)
        if no_split_modules is None:
            no_split_modules = []
        transformer_cls_names_to_wrap = list(no_split_modules)
        if fsdp2_plugin.transformer_cls_names_to_wrap is not None:
            transformer_cls_names_to_wrap = fsdp2_plugin.transformer_cls_names_to_wrap
        transformer_cls_to_wrap = set()

        for layer_class in transformer_cls_names_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
            transformer_cls_to_wrap.add(transformer_cls)

        def policy(module: torch.nn.Module) -> bool:
            if fsdp2_plugin.transformer_cls_names_to_wrap is None:
                return False
            return isinstance(module, tuple(transformer_cls_to_wrap))

    elif fn is size_based_auto_wrap_policy:

        def policy(module: torch.nn.Module) -> bool:
            module_num_params = sum(p.numel() for p in module.parameters())
            return module_num_params > fsdp2_plugin.min_num_params
    else:
        return None

    return policy


def get_fsdp2_grad_scaler(**kwargs):
    """
    Returns a `GradScaler` for FSDP2, as the current implementation of `get_grad_scaler` doesn't accept other args. We
    need this as current `get_grad_scaler` accepts only `distributed_type` as arg, which doesn't differentiate between
    FSDP1 and FSDP2
    """
    from torch.amp.grad_scaler import GradScaler

    return GradScaler(**kwargs)


def fsdp2_canonicalize_names(named_params: dict) -> dict:
    """Removes parameter name modifiers in order to map them back to their original names.

    See huggingface/accelerate#3554 for more context.

    Args:
        named_params (`dict`): The named parameters dictionary to canonicalize.

    Returns:
        `dict`: The canonicalized named parameters dictionary
    """
    named_params = {k.replace("._checkpoint_wrapped_module", ""): v for k, v in named_params.items()}
    named_params = {
        k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k: v for k, v in named_params.items()
    }
    named_params = {k.replace("._orig_mod", ""): v for k, v in named_params.items()}
    return named_params
