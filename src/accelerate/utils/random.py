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

import random
from typing import Optional, Union

import numpy as np
import torch

from ..state import AcceleratorState
from .constants import CUDA_DISTRIBUTED_TYPES
from .dataclasses import DistributedType, RNGType
from .imports import (
    get_torch_accelerator_backend,
    is_torch_xla_available,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


def set_seed(seed: int, device_specific: bool = False, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    if device_specific:
        seed += AcceleratorState().process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seeding can happen before a notebook launcher forks, so keep device discovery lazy.
    device_type, device_module = get_torch_accelerator_backend(check_available=False)
    # torch.manual_seed already seeds MPS, which has no manual_seed_all method.
    if device_type not in (None, "mps"):
        device_module.manual_seed_all(seed)
    if is_torch_xla_available():
        xm.set_rng_state(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    rng_type = RNGType(rng_type)
    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_torch_xla_available(), "Can't synchronize XLA seeds as torch_xla is unavailable."
        rng_state = torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        rng_state = generator.get_state()
    else:
        device_type, device_module = get_torch_accelerator_backend()
        assert device_type == rng_type.value, f"Can't synchronize {rng_type.value.upper()} seeds without that device."
        rng_state = device_module.get_rng_state()

    # Broadcast the rng state from device 0 to other devices
    state = AcceleratorState()
    if state.distributed_type == DistributedType.XLA:
        rng_state = rng_state.to(xm.xla_device())
        xm.collective_broadcast([rng_state])
        xm.mark_step()
        rng_state = rng_state.cpu()
    elif (
        state.distributed_type in CUDA_DISTRIBUTED_TYPES
        or state.distributed_type == DistributedType.MULTI_MLU
        or state.distributed_type == DistributedType.MULTI_SDAA
        or state.distributed_type == DistributedType.MULTI_MUSA
        or state.distributed_type == DistributedType.MULTI_NPU
        or state.distributed_type == DistributedType.MULTI_XPU
        or state.distributed_type == DistributedType.MULTI_HPU
        or state.distributed_type == DistributedType.MULTI_NEURON
    ):
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)

    # Set the broadcast rng state.
    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.XLA:
        xm.set_rng_state(rng_state.item())
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)
    else:
        device_module.set_rng_state(rng_state)


def synchronize_rng_states(rng_types: list[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)
