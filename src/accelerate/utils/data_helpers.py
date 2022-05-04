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

from typing import Mapping

import torch

from .dataclasses import TensorInformation
from .operations import is_tensor_information, is_torch_tensor


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))


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
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def send_to_device(tensor, device):
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

    def _send_to_device(t, device):
        return t.to(device)

    def _has_to_method(t):
        return hasattr(t, "to")

    return recursively_apply(_send_to_device, tensor, device, test_type=_has_to_method)


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
