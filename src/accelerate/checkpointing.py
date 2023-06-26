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

import collections
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from .utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    get_pretty_name,
    is_safetensors_available,
    is_tpu_available,
    is_xpu_available,
    save,
    dtype_byte_size,
    convert_file_size_to_int
)


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_safetensors_available():
    from safetensors.torch import storage_ptr, storage_size

import logging

from .logging import get_logger
from .state import PartialState


logger = get_logger(__name__)

logger_simple = logging.getLogger(__name__)


def save_accelerator_state(
    output_dir: str,
    model_states: List[dict],
    optimizers: list,
    schedulers: list,
    process_index: int,
    scaler: GradScaler = None,
):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (`List[torch.nn.Module]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save
    """
    # Model states
    for i, state in enumerate(model_states):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        output_model_file = os.path.join(output_dir, weights_name)
        save(state, output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")
    # Optimizer states
    for i, opt in enumerate(optimizers):
        state = opt.state_dict()
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        output_optimizer_file = os.path.join(output_dir, optimizer_name)
        save(state, output_optimizer_file)
        logger.info(f"Optimizer state saved in {output_optimizer_file}")
    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        state = scheduler.state_dict()
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        output_scheduler_file = os.path.join(output_dir, scheduler_name)
        save(state, output_scheduler_file)
        logger.info(f"Scheduler state saved in {output_scheduler_file}")
    # GradScaler state
    if scaler is not None:
        state = scaler.state_dict()
        output_scaler_file = os.path.join(output_dir, SCALER_NAME)
        torch.save(state, output_scaler_file)
        logger.info(f"Gradient scaler state saved in {output_scaler_file}")
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    if is_xpu_available():
        states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()
    else:
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    if is_tpu_available():
        states["xm_seed"] = xm.get_rng_state()
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")
    return output_dir


def load_accelerator_state(
    input_dir,
    models,
    optimizers,
    schedulers,
    process_index,
    scaler=None,
    map_location=None,
    **load_model_func_kwargs,
):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
        map_location (`str`, *optional*):
            What device to load the optimizer state onto. Should be one of either "cpu" or "on_device".
        load_model_func_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the model's `load_state_dict` method.
    """
    if map_location not in [None, "cpu", "on_device"]:
        raise TypeError(
            "Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`"
        )
    if map_location is None:
        map_location = "cpu"
    elif map_location == "on_device":
        map_location = PartialState().device
    # Model states
    for i, model in enumerate(models):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        input_model_file = os.path.join(input_dir, weights_name)
        models[i].load_state_dict(torch.load(input_model_file, map_location=map_location), **load_model_func_kwargs)
    logger.info("All model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        input_optimizer_file = os.path.join(input_dir, optimizer_name)
        optimizer_state = torch.load(input_optimizer_file, map_location=map_location)
        optimizers[i].load_state_dict(optimizer_state)
    logger.info("All optimizer states loaded successfully")

    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        input_scheduler_file = os.path.join(input_dir, scheduler_name)
        scheduler.load_state_dict(torch.load(input_scheduler_file))
    logger.info("All scheduler states loaded successfully")

    # GradScaler state
    if scaler is not None:
        input_scaler_file = os.path.join(input_dir, SCALER_NAME)
        scaler.load_state_dict(torch.load(input_scaler_file))
        logger.info("GradScaler state loaded successfully")

    # Random states
    try:
        states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
        else:
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        if is_tpu_available():
            xm.set_rng_state(states["xm_seed"])
        logger.info("All random states loaded successfully")
    except Exception:
        logger.info("Could not load random states")


def save_custom_state(obj, path, index: int = 0):
    """
    Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
    """
    # Should this be the right way to get a qual_name type value from `obj`?
    save_location = Path(path) / f"custom_checkpoint_{index}.pkl"
    logger.info(f"Saving the state of {get_pretty_name(obj)} to {save_location}")
    torch.save(obj.state_dict(), save_location)


def load_custom_state(obj, path, index: int = 0):
    """
    Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`
    """
    load_location = f"{path}/custom_checkpoint_{index}.pkl"
    logger.info(f"Loading the state of {get_pretty_name(obj)} from {load_location}")
    obj.load_state_dict(torch.load(load_location, map_location="cpu"))


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    return tensor.device, storage_ptr(tensor), storage_size(tensor)


def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_sahrd_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}

    for key, weight in state_dict.items():
        storage_id = id_tensor_storage(weight)

        # If a `weight` shares the same underlying storage as another tensor, we put `weight` in the same `block`
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if last_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def save_model(
    model: nn.Module,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    max_shard_size: Union[int, str] = "10GB",
    save_function: Callable = torch.save,
    safe_serialization: bool = False,
    variant: Optional[str] = None,
    keys_to_ignore_on_save: Optional[List[str]] = None,
):
    """
    Save an unwrapped model and its configuration file to a directory, so that it can be re-loaded using load_checkpoint_in_model

    Arguments:
        model: (`nn.Module`):
            Unwrapped model to be saved. You can unwarp your model using the `.unwrap_model(model)` method from Accelerator()
        save_directory (`str` or `os.PathLike`):
            Directory to which to save. Will be created if it doesn't exist.
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether the process calling this is the main process or not. Useful when in distributed training like TPUs
            and need to call this function on all processes. In this case, set `is_main_process=True` only on the main
            process to avoid race conditions.
        save_function (`Callable`):
            The function to use to save the state dictionary. Useful on distributed training like TPUs when one need to
            replace `torch.save` by another method.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower
            than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

            <Tip warning={true}>

            If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
            which will be bigger than `max_shard_size`.

            </Tip>

        safe_serialization (`bool`, *optional*, defaults to `False`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        variant (`str`, *optional*):
            If specified, weights are saved in the format pytorch_model.<variant>.bin.
        keys_to_ignore_on_save (`List[str]`, *optional*):
            The list of key that should not be saved
    """

    if safe_serialization and not is_safetensors_available():
        raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

    if os.path.isfile(save_directory):
        logger_simple.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return
    
    os.makedirs(save_directory, exist_ok=True)

    # Save the model
    state_dict = model.state_dict()

    # Handle the case where some state_dict keys shouldn't be saved
    if keys_to_ignore_on_save is not None:
        for ignore_key in keys_to_ignore_on_save:
            if ignore_key in state_dict.keys():
                del state_dict[ignore_key]
    if safe_serialization:
        # Safetensors does not allow tensor aliasing.
        # We're going to remove aliases before saving
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            ptrs[id_tensor_storage(tensor)].append(name)

        # These are all the pointers of shared tensors.
        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
        warn_names = set()
        for names in shared_ptrs.values():
            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            found = 0
            for name in names:
                if name in state_dict:
                    found += 1
                    if found > 1:
                        del state_dict[name]
                        warn_names.add(name)
        if len(warn_names) > 0:
            logger_simple.warning_once(
                f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
            )

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    weights_name = _add_variant(weights_name, variant)

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
            and is_main_process
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    for shard_file, shard in shards.items():
        save_function(shard, os.path.join(save_directory, shard_file))

    if index is None:
        path_to_weights = os.path.join(save_directory, _add_variant(WEIGHTS_NAME, variant))
        logger_simple.info(f"Model weights saved in {path_to_weights}")
    else:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger_simple.info(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )
