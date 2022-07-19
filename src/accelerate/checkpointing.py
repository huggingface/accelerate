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

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from .utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    get_pretty_name,
    is_tpu_available,
    save,
)
from .utils.dataclasses import DistributedType
from .utils.deepspeed import DeepSpeedSchedulerWrapper


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

from .logging import get_logger


logger = get_logger(__name__)


def save_accelerator_state(
    accelerator,
    output_dir: str,
    models: list,
    optimizers: list,
    schedulers: list,
    process_index: int,
    scaler: GradScaler = None,
):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    Args:
        accelerator (`Accelerator`):
            The Accelerator instance
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of models
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
    for i, model in enumerate(models):
        if accelerator.distributed_type == DistributedType.FSDP:
            logger.info("Saving FSDP model")
            accelerator.state.fsdp_plugin.save_model(accelerator, model, output_dir, i)
            logger.info(f"FSDP Model saved to output dir {output_dir}")
        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.info("Saving DeepSpeed Model and Optimizer")
            ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
            model.save_checkpoint(output_dir, ckpt_id)
            logger.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, ckpt_id)}")
        else:
            state = accelerator.get_state_dict(model, unwrap=False)
            weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            save(state, output_model_file)
            logger.info(f"Model weights saved in {output_model_file}")
    # Optimizer states
    for i, opt in enumerate(optimizers):
        if accelerator.distributed_type == DistributedType.FSDP:
            logger.info("Saving FSDP Optimizer")
            accelerator.state.fsdp_plugin.save_optimizer(accelerator, opt, models[i], output_dir, i)
            logger.info(f"FSDP Optimizer saved to output dir {output_dir}")
        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
            continue
        else:
            state = opt.state_dict()
            optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
            output_optimizer_file = os.path.join(output_dir, optimizer_name)
            save(state, output_optimizer_file)
            logger.info(f"Optimizer state saved in {output_optimizer_file}")
    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        if isinstance(scheduler, DeepSpeedSchedulerWrapper):
            continue
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
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    # ^^ safe to call this function even if cuda is not available
    if is_tpu_available():
        states["xm_seed"] = xm.get_rng_state()
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")
    return output_dir


def load_accelerator_state(
    accelerator,
    input_dir: str,
    models: list,
    optimizers: list,
    schedulers: list,
    process_index: int,
    scaler: GradScaler = None,
):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        accelerator (`Accelerator`):
            The Accelerator instance
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
    """
    # Model states
    for i, model in enumerate(models):
        if accelerator.distributed_type == DistributedType.FSDP:
            logger.info("Loading FSDP model")
            accelerator.state.fsdp_plugin.load_model(accelerator, model, input_dir, i)
            logger.info(f"FSDP Model loaded from input dir {input_dir}")
        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.info("Loading DeepSpeed Model and Optimizer")
            ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
            model.load_checkpoint(input_dir, ckpt_id)
            logger.info(f"DeepSpeed Model and Optimizer loaded from input dir {os.path.join(input_dir, ckpt_id)}")
        else:
            weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            models[i].load_state_dict(torch.load(input_model_file, map_location="cpu"))
            logger.info("model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        if accelerator.distributed_type == DistributedType.FSDP:
            logger.info("Loading FSDP Optimizer")
            accelerator.state.fsdp_plugin.load_optimizer(accelerator, opt, models[i], input_dir, i)
            logger.info(f"FSDP Optimizer loaded from input dir {input_dir}")
        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
            continue
        else:
            optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
            input_optimizer_file = os.path.join(input_dir, optimizer_name)
            optimizers[i].load_state_dict(torch.load(input_optimizer_file, map_location="cpu"))
            logger.info("optimizer states loaded successfully")

    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        if isinstance(scheduler, DeepSpeedSchedulerWrapper):
            continue
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
    states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
    random.setstate(states["random_state"])
    np.random.set_state(states["numpy_random_seed"])
    torch.set_rng_state(states["torch_manual_seed"])
    torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    # ^^ safe to call this function even if cuda is not available
    if is_tpu_available():
        xm.set_rng_state(states["xm_seed"])
    logger.info("All random states loaded successfully")


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
    obj.load_state_dict(torch.load(load_location))
