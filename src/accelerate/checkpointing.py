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

import os
import random

import numpy as np
import torch

from .state import is_tpu_available
from .utils import MODEL_NAME, OPTIMIZER_NAME, RNG_STATE_NAME, SCALAR_NAME, save


if is_tpu_available():
    import torch_xla.core.xla_model as xm

import logging


logger = logging.getLogger(__name__)


def save_accelerator_state(model_states, optimizers, process_index, scalar=None, output_dir="."):
    """
    Saves the current states of the `models`, `optimizers`, scalar, and RNG generators to `output_dir`.

    Args:
        model_states (:obj:`list`):
            A list of model state dicts received from `Accelerator().get_state_dict`
        output_dir (:obj:`str` or :obj:`os.PathLike`):
            The name of the folder to save all relevant weights and states.
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
        torch.save(state, output_optimizer_file)
        logger.info(f"Optimizer state saved in {output_optimizer_file}")
    # GradScalar state
    if scaler is not None:
        state = scaler.state_dict()
        output_scaler_file = os.path.join(output_dir, SCALAR_NAME)
        torch.save(state, output_scaler_file)
        logger.info(f"Gradient scaler state saved in {output_scaler_file}")
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state()
    # ^^ safe to call this function even if cuda is not available
    if is_tpu_available():
        states["xm_seed"] = torch.tensor(xm.get_rng_state())
    output_states_file = os.path.join(output_dir, states_name)
    torch.save(states, output_states_file)
    logger.info(f"Random states saved in {output_states_file}")
    return output_dir


# def load_accelerator_state(models, optimizers, scalar, input_dir)
