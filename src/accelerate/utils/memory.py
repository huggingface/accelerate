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
import inspect
import os
import sys
from typing import Optional

import torch

from .environment import str_to_bool
from .imports import (
    is_cuda_available,
    is_hpu_available,
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_neuron_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)


# Default for capping the CPU budget of `get_max_memory` at the cgroup memory limit of the process. Flip this one
# line to change the default; `ACCELERATE_RESPECT_CGROUP_MEMORY=1` or `=0` overrides it.
RESPECT_CGROUP_MEMORY_BY_DEFAULT = False

_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_PROC_SELF_MOUNTINFO = "/proc/self/mountinfo"
# cgroup v1 reports "no limit" as a huge sentinel instead of `max`.
_CGROUP_UNLIMITED = 1 << 62


def _read_cgroup_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _read_memory_stat(path: str, key: str) -> int:
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _path_parts(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _cgroup_membership(controller: Optional[str]) -> str:
    """The cgroup path of this process from `/proc/self/cgroup`, relative to the hierarchy root."""
    try:
        with open(_PROC_SELF_CGROUP) as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                is_v2 = controller is None and parts[0] == "0" and parts[1] == ""
                is_v1 = controller is not None and controller in parts[1].split(",")
                if is_v2 or is_v1:
                    return parts[2]
    except OSError:
        pass
    return ""


def _cgroup_mount(controller: Optional[str], membership: str) -> tuple[str, str]:
    """
    Returns (mount point, mount root) of the cgroup hierarchy from `/proc/self/mountinfo`.

    The mount can expose a subtree of the hierarchy (mount root other than `/`, e.g. Docker with the host cgroup
    namespace), cgroup v1 can co-mount controllers under a name such as `cpu,memory`, and several mounts can match,
    e.g. a bind mount of the container subtree next to the full hierarchy. The mount whose root contains the
    membership is used; among those, the one with the shortest root, because it exposes the most ancestors and an
    ancestor limit still applies. Then the first match; then `/sys/fs/cgroup`.
    """
    candidates = []
    try:
        with open(_PROC_SELF_MOUNTINFO) as f:
            for line in f:
                before, separator, after = line.partition(" - ")
                mount_fields, fs_fields = before.split(), after.split()
                if not separator or len(mount_fields) < 5 or len(fs_fields) < 3:
                    continue
                fs_type, super_options = fs_fields[0], fs_fields[2].split(",")
                is_v2 = controller is None and fs_type == "cgroup2"
                is_v1 = controller is not None and fs_type == "cgroup" and controller in super_options
                if is_v2 or is_v1:
                    # mountinfo escapes spaces as \040.
                    candidates.append((mount_fields[4].replace("\\040", " "), mount_fields[3].replace("\\040", " ")))
    except OSError:
        pass
    member_parts = _path_parts(membership)
    containing = [
        (mount_point, mount_root)
        for mount_point, mount_root in candidates
        if member_parts[: len(_path_parts(mount_root))] == _path_parts(mount_root)
    ]
    if containing:
        return min(containing, key=lambda mount: len(_path_parts(mount[1])))
    if candidates:
        return candidates[0]
    return (_CGROUP_ROOT if controller is None else os.path.join(_CGROUP_ROOT, controller)), "/"


def _cgroup_dirs(controller: Optional[str]) -> list[str]:
    """Directories of this process's cgroup, nearest first, mount point last."""
    membership = _cgroup_membership(controller)
    mount_point, mount_root = _cgroup_mount(controller, membership)
    components = _path_parts(membership)
    root_components = _path_parts(mount_root)
    if ".." in components:
        # A cgroup outside this cgroup namespace cannot be resolved from here.
        components = []
    elif components[: len(root_components)] == root_components:
        # /proc/self/cgroup is relative to the hierarchy root; the mount starts at mount_root.
        components = components[len(root_components) :]
    dirs = [os.path.join(mount_point, *components[:depth]) for depth in range(len(components), 0, -1)]
    return dirs + [mount_point]


def get_cgroup_available_memory() -> Optional[int]:
    """
    Returns the memory in bytes that the cgroup of this process can still use, or `None` when no cgroup memory limit
    applies (or the check is disabled, see `RESPECT_CGROUP_MEMORY_BY_DEFAULT`).

    Every cgroup from the process's own one up to the cgroup root can carry a limit, and the kernel enforces all of
    them, so the result is the smallest headroom (limit minus usage) over every limited level: a parent with a larger
    limit can still have less room when its other children use memory. Usage excludes the inactive page cache, which
    the kernel can reclaim. Returns `None` when no limited level exposes its usage.
    """
    override = os.environ.get("ACCELERATE_RESPECT_CGROUP_MEMORY")
    enabled = RESPECT_CGROUP_MEMORY_BY_DEFAULT if override is None else str_to_bool(override, to_bool=True)
    if not enabled or sys.platform != "linux":
        return None
    for controller, limit_file, usage_file, inactive_key in (
        (None, "memory.max", "memory.current", "inactive_file"),
        ("memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"),
    ):
        limited = False
        available = None
        for cgroup_dir in _cgroup_dirs(controller):
            limit = _read_cgroup_int(os.path.join(cgroup_dir, limit_file))
            if limit is None or limit >= _CGROUP_UNLIMITED:
                continue
            limited = True
            usage = _read_cgroup_int(os.path.join(cgroup_dir, usage_file))
            if usage is None:
                continue
            inactive_file = _read_memory_stat(os.path.join(cgroup_dir, "memory.stat"), inactive_key)
            headroom = max(0, limit - max(0, usage - inactive_file))
            available = headroom if available is None else min(available, headroom)
        if limited:
            return available
    return None


def clear_device_cache(garbage_collection: bool = False) -> None:
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
    elif is_neuron_available():
        # Not sure it actually does something, but adding for consistency with other backends
        torch.neuron.empty_cache()


def release_memory(*objects) -> list:
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
    Checks if `exception` relates to CUDA, XPU, MPS or CPU out-of-memory, or CUDNN not supported

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        " out of memory.",  # OOM for CUDA, HIP, XPU
        "MPS backend out of memory",  # MPS OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        "FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed",  # HPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(
    function: Optional[callable] = None,
    starting_batch_size: int = 128,
    reduce_batch_size_fn: Optional[callable] = None,
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
        reduce_batch_size_fn (`callable`, *optional*):
            A function to determine the new batch size after an out-of-memory error. If not
            provided, the batch size is multiplied by 0.9 on each failure. The function takes
            no arguments and should return the new (reduced) batch size as an `int`.

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
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            reduce_batch_size_fn=reduce_batch_size_fn,
        )

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
