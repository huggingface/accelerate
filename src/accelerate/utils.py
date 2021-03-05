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

import random

import numpy as np
import torch

from .state import AcceleratorState, DistributedType, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def synchronize_rng_states():
    """
    Helper function to synchronize the rng states in distributed training.
    """
    state = AcceleratorState()
    if state.distributed_type == DistributedType.TPU:
        rng_state = torch.get_rng_state()
        rng_state = xm.mesh_reduce("random_seed", rng_state, lambda x: x[0])
        torch.set_rng_state(rng_state)
    elif state.distributed_type == DistributedType.MULTI_GPU:
        rng_state = torch.get_rng_state().to(state.device)
        # Broadcast the state from process 0 to all the others.
        torch.distributed.broadcast(rng_state, 0)
        torch.set_rng_state(rng_state.cpu())

        # Broadcast the state from process 0 to all the others.
        rng_state = torch.cuda.get_rng_state().to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        torch.cuda.set_rng_state(rng_state.cpu())


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        raise TypeError(
            f"Can't send the values of type {type(tensor)} to device {device}, only of nested list/tuple/dicts "
            "of tensors or objects having a `to` method."
        )
    return tensor.to(device)


def extract_model_from_parallel(model):
    """
    Extract a model from its distributed containers.

    Args:
        model (:obj:`torch.nn.Module`): The model to extract.

    Returns:
        :obj:`torch.nn.Module`: The extracted model.
    """
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def _tpu_gather(tensor, name="tensor"):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_tpu_gather(t, name=f"{name}_{i}") for i, t in enumerate(tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _tpu_gather(v, name=f"{name}_{k}") for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    return xm.mesh_reduce(name, tensor, torch.cat)


def _gpu_gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_gpu_gather(t) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _gpu_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


def gather(tensor):
    """
    Recusrively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor, name="accelerate.utils.gather")
    elif AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        return _gpu_gather(tensor)
    else:
        return tensor


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    Warning::

        Make sure all processes will reach this instruction otherwise one of your processes will hang forever.
    """
    if AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        torch.distributed.barrier()
    elif AcceleratorState().distributed_type == DistributedType.TPU:
        xm.rendezvous("accelerate.utils.wait_for_everyone")


def save(obj, f):
    """
    Save the data to disk. Use in place of :obj:`torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        xm.save(obj, f)
    elif AcceleratorState().local_process_index == 0:
        torch.save(obj, f)
