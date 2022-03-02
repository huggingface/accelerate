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
from typing import List

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from .state import is_tpu_available
from .utils import MODEL_NAME, OPTIMIZER_NAME, RNG_STATE_NAME, SCALER_NAME, save


if is_tpu_available():
    import torch_xla.core.xla_model as xm

import logging


logger = logging.getLogger(__name__)


def save_accelerator_state(
    output_dir: str, model_states: List[dict], optimizers: list, process_index: int, scaler: GradScaler = None
):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    Args:
        output_dir (:obj:`str` or :obj:`os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (:obj:`List[torch.nn.Module]`):
            A list of model states
        optimizers (:obj:`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        process_index (:obj:`int`):
            The current process index in the Accelerator state
        scaler (:obj:`torch.cuda.amp.GradScaler`, `optional`):
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
        states["xm_seed"] = torch.tensor(xm.get_rng_state())
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")
    return output_dir


def load_accelerator_state(input_dir, models, optimizers, process_index, scaler=None):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (:obj:`str` or :obj:`os.PathLike`):
            The name of the folder to load all relevant weights and states.
        model_stmodelsates (:obj:`List[torch.nn.Module]`):
            A list of model instances
        optimizers (:obj:`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        process_index (:obj:`int`):
            The current process index in the Accelerator state
        scaler (:obj:`torch.cuda.amp.GradScaler`, `optional`):
            An optional `GradScaler` instance to load
    """
    # Model states
    for i, model in enumerate(models):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        input_model_file = os.path.join(input_dir, weights_name)
        models[i].load_state_dict(torch.load(input_model_file))
    logger.info("All model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        input_optimizer_file = os.path.join(input_dir, optimizer_name)
        optimizers[i].load_state_dict(torch.load(input_optimizer_file))
    logger.info("All optimizer states loaded successfully")

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
