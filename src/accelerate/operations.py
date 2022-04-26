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


from typing import Any, Mapping

import torch
from torch.distributed import ReduceOp

from .state import AcceleratorState, DistributedType, is_tpu_available
from .utils import honor_type, recursively_apply


if is_tpu_available():
    import torch_xla.core.xla_model as xm


def _tpu_gather(tensor, name="gather tensor"):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_tpu_gather(t, name=f"{name}_{i}") for i, t in enumerate(tensor)))
    elif isinstance(tensor, Mapping):
        return type(tensor)({k: _tpu_gather(v, name=f"{name}_{k}") for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    if tensor.ndim == 0:
        tensor = tensor.clone()[None]
    return xm.mesh_reduce(name, tensor, torch.cat)


def _gpu_gather(tensor):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


_cpu_gather = _gpu_gather


def gather(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor, name="accelerate.utils.gather")
    elif AcceleratorState().distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
        return _gpu_gather(tensor)
    elif AcceleratorState().distributed_type == DistributedType.MULTI_CPU:
        return _cpu_gather(tensor)
    else:
        return tensor


def _gpu_gather_object(object: Any):
    def _gpu_gather_object_one(object: Any):
        output_objects = [None for _ in range(AcceleratorState().num_processes)]
        torch.distributed.all_gather_object(output_objects, object)
        return output_objects

    return recursively_apply(_gpu_gather_object_one, object)


_cpu_gather_object = _gpu_gather_object


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        raise NotImplementedError("gather objects in TPU is not supported")
    elif AcceleratorState().distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
        return _gpu_gather_object(object)
    elif AcceleratorState().distributed_type == DistributedType.MULTI_CPU:
        return _cpu_gather_object(object)
    else:
        return object


def _gpu_broadcast(data, src=0):
    def _gpu_broadcast_one(tensor, src=0):
        torch.distributed.broadcast(tensor, src=src)
        return tensor

    return recursively_apply(_gpu_broadcast_one, data, error_on_other_type=True, src=src)


def _tpu_broadcast(tensor, src=0, name="broadcast tensor"):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_tpu_broadcast(t, name=f"{name}_{i}") for i, t in enumerate(tensor)))
    elif isinstance(tensor, Mapping):
        return type(tensor)({k: _tpu_broadcast(v, name=f"{name}_{k}") for k, v in tensor.items()})
    return xm.mesh_reduce(name, tensor, lambda x: x[src])


def broadcast(tensor, from_process: int = 0):
    """
    Recursively broadcast tensor in a nested list/tuple/dictionary of tensors to all devices.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.
        from_process (`int`, *optional*, defaults to 0):
            The process from which to send the data

    Returns:
        The same data structure as `tensor` with all tensors broadcasted to the proper device.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        return _tpu_broadcast(tensor, src=from_process, name="accelerate.utils.broadcast")
    elif AcceleratorState().distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
        return _gpu_broadcast(tensor, src=from_process)
    elif AcceleratorState().distributed_type == DistributedType.MULTI_CPU:
        return _gpu_broadcast(tensor, src=from_process)
    else:
        return tensor


def broadcast_object_list(object_list, from_process: int = 0):
    """
    Broadcast a list of picklable objects form one process to the others.

    Args:
        object_list (list of picklable objects):
            The list of objects to broadcast. This list will be modified inplace.
        from_process (`int`, *optional*, defaults to 0):
            The process from which to send the data.

    Returns:
        The same list containing the objects from process 0.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        for i, obj in enumerate(object_list):
            object_list[i] = xm.mesh_reduce("accelerate.utils.broadcast_object_list", obj, lambda x: x[from_process])
    elif AcceleratorState().distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
        torch.distributed.broadcast_object_list(object_list, src=from_process)
    elif AcceleratorState().distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast_object_list(object_list, src=from_process)
    return object_list


def slice_tensors(data, tensor_slice):
    """
    Recursively takes a slice in a nested list/tuple/dictionary of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`):
            The data to slice.
        tensor_slice (`slice`):
            The slice to take.

    Returns:
        The same data structure as `data` with all the tensors slices.
    """

    def _slice_tensor(tensor, tensor_slice):
        return tensor[tensor_slice]

    return recursively_apply(_slice_tensor, data, tensor_slice)


def concatenate(data, dim=0):
    """
    Recursively concatenate the tensors in a nested list/tuple/dictionary of lists of tensors with the same shape.

    Args:
        data (nested list/tuple/dictionary of lists of tensors `torch.Tensor`):
            The data to concatenate.
        dim (`int`, *optional*, defaults to 0):
            The dimension on which to concatenate.

    Returns:
        The same data structure as `data` with all the tensors concatenated.
    """
    if isinstance(data[0], (tuple, list)):
        return honor_type(data[0], (concatenate([d[i] for d in data], dim=dim) for i in range(len(data[0]))))
    elif isinstance(data[0], Mapping):
        return type(data[0])({k: concatenate([d[k] for d in data], dim=dim) for k in data[0].keys()})
    elif not isinstance(data[0], torch.Tensor):
        raise TypeError(f"Can only concatenate tensors but got {type(data[0])}")
    return torch.cat(data, dim=dim)


def pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so they
    can safely be gathered.

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

    def _pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
        if dim >= len(tensor.shape):
            return tensor

        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = gather(size).cpu()
        # Then pad to the maximum size
        max_size = max(s[dim] for s in sizes)
        if max_size == tensor.shape[dim]:
            return tensor

        old_size = tensor.shape
        new_size = list(old_size)
        new_size[dim] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        if pad_first:
            indices = tuple(
                slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))
            )
        else:
            indices = tuple(slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size)))
        new_tensor[indices] = tensor
        return new_tensor

    return recursively_apply(
        _pad_across_processes, tensor, error_on_other_type=True, dim=dim, pad_index=pad_index, pad_first=pad_first
    )


def reduce(tensor: torch.Tensor, reduction="mean"):
    """
    Reduce `tensor` across all processes and perform a `reduction`.

    Args:
        tensor (`torch.Tensor`):
            The data to reduce
        reduction (`str`, *optional*, defaults to "mean"):
            A reduction method. Can be of "mean", "sum", or "none"
    """
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(f"Invalid `reduction` passed: {reduction}." "Must be either 'mean', 'sum', or 'none'.")
    if reduction == "none":
        return tensor
    state = AcceleratorState()
    cloned_tensor = tensor.clone()
    if state.distributed_type == DistributedType.TPU:
        xm.all_reduce("sum", cloned_tensor)
    elif state.distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
        cloned_tensor = tensor.clone()
        torch.distributed.reduce(cloned_tensor, ReduceOp.SUM)
    else:
        if reduction == "sum":
            return cloned_tensor.sum()
        else:
            return cloned_tensor.mean()
