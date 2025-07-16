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
A collection of utilities for ensuring that training can always occur. Heavily influenced by the
[toma](https://github.com/BlackHC/toma) library.
"""

import functools
import gc
import importlib
import inspect
import warnings

import torch
from packaging import version

from .imports import (
    is_cuda_available,
    is_hpu_available,
    is_ipex_available,
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from .versions import compare_versions


def clear_device_cache(garbage_collection=False):
    """
    Clears the device cache by calling `torch.{backend}.empty_cache`. Can also run `gc.collect()`, but do note that
    this is a *considerable* slowdown and should be used sparingly.
    """
    if garbage_collection:
        gc.collect()

    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif is_mps_available(min_version="2.0"):
        torch.mps.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()
    elif is_hpu_available():
        # torch.hpu.empty_cache() # not available on hpu as it reserves all device memory for the current process
        pass


def release_memory(*objects):
    """
    Releases memory from `objects` by setting them to `None` and calls `gc.collect()` and `torch.cuda.empty_cache()`.
    Returned objects should be reassigned to the same variables.

    Args:
        objects (`Iterable`):
            An iterable of objects
    Returns:
        A list of `None` objects to replace `objects`

    Example:

        ```python
        >>> import torch
        >>> from accelerate.utils import release_memory

        >>> a = torch.ones(1000, 1000).cuda()
        >>> b = torch.ones(1000, 1000).cuda()
        >>> a, b = release_memory(a, b)
        ```
    """
    if not isinstance(objects, list):
        objects = list(objects)
    for i in range(len(objects)):
        objects[i] = None
    clear_device_cache(garbage_collection=True)
    return objects


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, XPU out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        " out of memory.",  # OOM for CUDA, HIP, XPU
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        "FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed",  # HPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128, reduce_batch_size_fn: callable = None
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is multiplied by 0.9 and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from accelerate.utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size
    if reduce_batch_size_fn is None:

        def reduce_batch_size_fn():
            nonlocal batch_size
            batch_size = int(batch_size * 0.9)
            return batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        clear_device_cache(garbage_collection=True)
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    clear_device_cache(garbage_collection=True)
                    batch_size = reduce_batch_size_fn()
                else:
                    raise

    return decorator


def get_xpu_available_memory(device_index: int):
    if version.parse(torch.__version__).release >= version.parse("2.6").release:
        # torch.xpu.mem_get_info API is available starting from PyTorch 2.6
        # It further requires PyTorch built with the SYCL runtime which supports API
        # to query available device memory. If not available, exception will be
        # raised. Version of SYCL runtime used to build PyTorch is being reported
        # with print(torch.version.xpu) and corresponds to the version of Intel DPC++
        # SYCL compiler. First version to support required feature is 20250001.
        try:
            return torch.xpu.mem_get_info(device_index)[0]
        except Exception:
            pass
    elif is_ipex_available():
        ipex_version = version.parse(importlib.metadata.version("intel_extension_for_pytorch"))
        if compare_versions(ipex_version, ">=", "2.5"):
            from intel_extension_for_pytorch.xpu import mem_get_info

            return mem_get_info(device_index)[0]

    warnings.warn(
        "The XPU `mem_get_info` API is available in IPEX version >=2.5 or PyTorch >=2.6. The current returned available memory is incorrect. Please consider upgrading your IPEX or PyTorch version."
    )
    return torch.xpu.max_memory_allocated(device_index)
