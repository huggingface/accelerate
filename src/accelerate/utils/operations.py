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

"""
A set of basic tensor ops compatible with tpu, gpu, and multigpu
"""

import pickle
from functools import update_wrapper, wraps
from typing import Any, Mapping

import torch

from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import is_torch_distributed_available, is_tpu_available


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


if is_torch_distributed_available():
    from torch.distributed import ReduceOp


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_torch_xpu_tensor(tensor):
    return isinstance(
        tensor,
        torch.xpu.FloatTensor,
        torch.xpu.ByteTensor,
        torch.xpu.IntTensor,
        torch.xpu.LongTensor,
        torch.xpu.HalfTensor,
        torch.xpu.DoubleTensor,
        torch.xpu.BFloat16Tensor,
    )


def is_tensor_information(tensor_info):
    return isinstance(tensor_info, TensorInformation)


def is_namedtuple(data):
    """
    Checks if `x` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    data_type = type(data)
    bases = data_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(data_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(member, str) for member in fields)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs:
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor)
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for k, t in tensor.items()
            }
        )
    elif hasattr(tensor, "to"):
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
    else:
        return tensor


def get_data_structure(data):
    """
    Recursively gathers the information needed to rebuild a nested list/tuple/dictionary of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to analyze.

    Returns:
        The same data structure as `data` with [`~utils.TensorInformation`] instead of tensors.
    """

    def _get_data_structure(tensor):
        return TensorInformation(shape=tensor.shape, dtype=tensor.dtype)

    return recursively_apply(_get_data_structure, data)


def get_shape(data):
    """
    Recursively gathers the shape of a nested list/tuple/dictionary of tensors as a list.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to analyze.

    Returns:
        The same data structure as `data` with lists of tensor shapes instead of tensors.
    """

    def _get_shape(tensor):
        return list(tensor.shape)

    return recursively_apply(_get_shape, data)


def initialize_tensors(data_structure):
    """
    Recursively initializes tensors from a nested list/tuple/dictionary of [`~utils.TensorInformation`].

    Returns:
        The same data structure as `data` with tensors instead of [`~utils.TensorInformation`].
    """

    def _initialize_tensor(tensor_info):
        return torch.empty(*tensor_info.shape, dtype=tensor_info.dtype)

    return recursively_apply(_initialize_tensor, data_structure, test_type=is_tensor_information)


def find_batch_size(data):
    """
    Recursively finds the batch size in a nested list/tuple/dictionary of lists of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to find the batch size.

    Returns:
        `int`: The batch size.
    """
    if isinstance(data, (tuple, list)):
        return find_batch_size(data[0])
    elif isinstance(data, Mapping):
        for k in data.keys():
            return find_batch_size(data[k])
    elif not isinstance(data, torch.Tensor):
        raise TypeError(f"Can only find the batch size of tensors but got {type(data)}.")
    return data.shape[0]


def listify(data):
    """
    Recursively finds tensors in a nested list/tuple/dictionary and converts them to a list of numbers.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to convert to regular numbers.

    Returns:
        The same data structure as `data` with lists of numbers instead of `torch.Tensor`.
    """

    def _convert_to_list(tensor):
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
            # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
            # Until Numpy adds bfloat16, we must convert float32.
            tensor = tensor.to(torch.float32)
        return tensor.tolist()

    return recursively_apply(_convert_to_list, data)


def _tpu_gather(tensor):
    def _tpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]

        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return xm.all_gather(tensor)

    res = recursively_apply(_tpu_gather_one, tensor, error_on_other_type=True)
    xm.mark_step()
    return res


def _gpu_gather(tensor):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]

        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        output_tensors = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


class DistributedOperationException(Exception):
    """
    An exception class for distributed operations. Raised if the operation cannot be performed due to the shape of the
    tensors.
    """

    pass


def verify_operation(function):
    """
    Verifies that `tensor` is the same shape across all processes. Only ran if `PartialState().debug` is `True`.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        if PartialState().distributed_type == DistributedType.NO or not PartialState().debug:
            return function(*args, **kwargs)
        operation = f"{function.__module__}.{function.__name__}"
        if "tensor" in kwargs:
            tensor = kwargs["tensor"]
        else:
            tensor = args[0]
        shapes = get_shape(tensor)
        output = gather_object([shapes])
        if output[0] is not None:
            are_same = output.count(output[0]) == len(output)
            if not are_same:
                process_shape_str = "\n  - ".join([f"Process {i}: {shape}" for i, shape in enumerate(output)])
                raise DistributedOperationException(
                    f"Cannot apply desired operation due to shape mismatches. "
                    "All shapes across devices must be valid."
                    f"\n\nOperation: `{operation}`\nInput shapes:\n  - {process_shape_str}"
                )
        return function(*args, **kwargs)

    return wrapper


def chained_operation(function):
    """
    Checks that `verify_operation` failed and if so reports a more helpful error chaining the existing
    `DistributedOperationException`.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except DistributedOperationException as e:
            operation = f"{function.__module__}.{function.__name__}"
            raise DistributedOperationException(
                f"Error found while calling `{operation}`. Please see the earlier error for more details."
            ) from e

    return wrapper


@verify_operation
def gather(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if PartialState().distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor)
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather(tensor)
    else:
        return tensor


def _gpu_gather_object(object: Any):
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.TPU:
        raise NotImplementedError("gather objects in TPU is not supported")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
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


@verify_operation
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
    if PartialState().distributed_type == DistributedType.TPU:
        return _tpu_broadcast(tensor, src=from_process, name="accelerate.utils.broadcast")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
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
    if PartialState().distributed_type == DistributedType.TPU:
        for i, obj in enumerate(object_list):
            object_list[i] = xm.mesh_reduce("accelerate.utils.broadcast_object_list", obj, lambda x: x[from_process])
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        torch.distributed.broadcast_object_list(object_list, src=from_process)
    return object_list


def slice_tensors(data, tensor_slice, process_index=None, num_processes=None):
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


@chained_operation
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


@verify_operation
def reduce(tensor, reduction="mean", scale=1.0):
    """
    Recursively reduce the tensors in a nested list/tuple/dictionary of lists of tensors across all processes by the
    mean of a given operation.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to reduce.
        reduction (`str`, *optional*, defaults to `"mean"`):
            A reduction method. Can be of "mean", "sum", or "none"
        scale (`float`, *optional*):
            A default scaling value to be applied after the reduce, only valied on XLA.

    Returns:
        The same data structure as `data` with all the tensors reduced.
    """

    def _reduce_across_processes(tensor, reduction="mean", scale=1.0):
        state = PartialState()
        cloned_tensor = tensor.clone()
        if state.distributed_type == DistributedType.NO:
            return cloned_tensor
        if state.distributed_type == DistributedType.TPU:
            xm.all_reduce("sum", cloned_tensor, scale)
        elif state.distributed_type.value in TORCH_DISTRIBUTED_OPERATION_TYPES:
            torch.distributed.all_reduce(cloned_tensor, ReduceOp.SUM)
        if reduction == "mean":
            cloned_tensor /= state.num_processes
        return cloned_tensor

    return recursively_apply(
        _reduce_across_processes, tensor, error_on_other_type=True, reduction=reduction, scale=scale
    )


def convert_to_fp32(tensor):
    """
    Recursively converts the elements nested list/tuple/dictionary of tensors in FP16/BF16 precision to FP32.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to convert from FP16/BF16 to FP32.

    Returns:
        The same data structure as `tensor` with all tensors that were in FP16/BF16 precision converted to FP32.
    """

    def _convert_to_fp32(tensor):
        return tensor.float()

    def _is_fp16_bf16_tensor(tensor):
        return hasattr(tensor, "dtype") and tensor.dtype in (torch.float16, torch.bfloat16)

    return recursively_apply(_convert_to_fp32, tensor, test_type=_is_fp16_bf16_tensor)


class ConvertOutputsToFp32:
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP16
    precision will be convert back to FP32.

    Args:
        model_forward (`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as `model_forward` but with converted outputs.
    """

    def __init__(self, model_forward):
        self.model_forward = model_forward
        update_wrapper(self, model_forward)

    def __call__(self, *args, **kwargs):
        return convert_to_fp32(self.model_forward(*args, **kwargs))

    def __getstate__(self):
        raise pickle.PicklingError(
            "Cannot pickle a prepared model with automatic mixed precision, please unwrap the model with `Accelerator.unwrap_model(model)` before pickling it."
        )


def convert_outputs_to_fp32(model_forward):
    model_forward = ConvertOutputsToFp32(model_forward)

    def forward(*args, **kwargs):
        return model_forward(*args, **kwargs)

    # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`
    forward.__wrapped__ = model_forward

    return forward


def find_device(data):
    """
    Finds the device on which a nested dict/list/tuple of tensors lies (assuming they are all on the same device).

    Args:
        (nested list/tuple/dictionary of `torch.Tensor`): The data we want to know the device of.
    """
    if isinstance(data, Mapping):
        for obj in data.values():
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, (tuple, list)):
        for obj in data:
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, torch.Tensor):
        return data.device
