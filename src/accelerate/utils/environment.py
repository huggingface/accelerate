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

import logging
import math
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional

import torch
from packaging.version import parse


logger = logging.getLogger(__name__)


def convert_dict_to_env_variables(current_env: dict):
    """
    Verifies that all keys and values in `current_env` do not contain illegal keys or values, and returns a list of
    strings as the result.

    Example:
    ```python
    >>> from accelerate.utils.environment import verify_env

    >>> env = {"ACCELERATE_DEBUG_MODE": "1", "BAD_ENV_NAME": "<mything", "OTHER_ENV": "2"}
    >>> valid_env_items = verify_env(env)
    >>> print(valid_env_items)
    ["ACCELERATE_DEBUG_MODE=1\n", "OTHER_ENV=2\n"]
    ```
    """
    forbidden_chars = [";", "\n", "<", ">", " "]
    valid_env_items = []
    for key, value in current_env.items():
        if all(char not in (key + value) for char in forbidden_chars) and len(key) >= 1 and len(value) >= 1:
            valid_env_items.append(f"{key}={value}\n")
        else:
            logger.warning(f"WARNING: Skipping {key}={value} as it contains forbidden characters or missing values.")
    return valid_env_items


def str_to_bool(value) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value}")


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def parse_flag_from_env(key, default=False):
    """Returns truthy value for `key` from the env if available else the default."""
    value = os.environ.get(key, str(default))
    return str_to_bool(value) == 1  # As its name indicates `str_to_bool` actually returns an int...


def parse_choice_from_env(key, default="no"):
    value = os.environ.get(key, str(default))
    return value


def are_libraries_initialized(*library_names: str) -> List[str]:
    """
    Checks if any of `library_names` are imported in the environment. Will return any names that are.
    """
    return [lib_name for lib_name in library_names if lib_name in sys.modules.keys()]


def _nvidia_smi():
    """
    Returns the right nvidia-smi command based on the system.
    """
    if platform.system() == "Windows":
        # If platform is Windows and nvidia-smi can't be found in path
        # try from systemd drive with default installation path
        command = which("nvidia-smi")
        if command is None:
            command = f"{os.environ['systemdrive']}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
    else:
        command = "nvidia-smi"
    return command


def get_gpu_info():
    """
    Gets GPU count and names using `nvidia-smi` instead of torch to not initialize CUDA.

    Largely based on the `gputil` library.
    """
    # Returns as list of `n` GPUs and their names
    output = subprocess.check_output(
        [_nvidia_smi(), "--query-gpu=count,name", "--format=csv,noheader"], universal_newlines=True
    )
    output = output.strip()
    gpus = output.split(os.linesep)
    # Get names from output
    gpu_count = len(gpus)
    gpu_names = [gpu.split(",")[1].strip() for gpu in gpus]
    return gpu_names, gpu_count


def get_driver_version():
    """
    Returns the driver version

    In the case of multiple GPUs, will return the first.
    """
    output = subprocess.check_output(
        [_nvidia_smi(), "--query-gpu=driver_version", "--format=csv,noheader"], universal_newlines=True
    )
    output = output.strip()
    return output.split(os.linesep)[0]


def check_cuda_p2p_ib_support():
    """
    Checks if the devices being used have issues with P2P and IB communications, namely any consumer GPU hardware after
    the 3090.

    Noteably uses `nvidia-smi` instead of torch to not initialize CUDA.
    """
    try:
        device_names, device_count = get_gpu_info()
        # As new consumer GPUs get released, add them to `unsupported_devices``
        unsupported_devices = {"RTX 40"}
        if device_count > 1:
            if any(
                unsupported_device in device_name
                for device_name in device_names
                for unsupported_device in unsupported_devices
            ):
                # Check if they have the right driver version
                acceptable_driver_version = "550.40.07"
                current_driver_version = get_driver_version()
                if parse(current_driver_version) < parse(acceptable_driver_version):
                    return False
                return True
    except Exception:
        pass
    return True


def check_fp8_capability():
    """
    Checks if all the current GPUs available support FP8.

    Notably must initialize `torch.cuda` to check.
    """
    cuda_device_capacity = torch.cuda.get_device_capability()
    return cuda_device_capacity >= (8, 9)


@dataclass
class CPUInformation:
    """
    Stores information about the CPU in a distributed environment. It contains the following attributes:
    - rank: The rank of the current process.
    - world_size: The total number of processes in the world.
    - local_rank: The rank of the current process on the local node.
    - local_world_size: The total number of processes on the local node.
    """

    rank: int = field(default=0, metadata={"help": "The rank of the current process."})
    world_size: int = field(default=1, metadata={"help": "The total number of processes in the world."})
    local_rank: int = field(default=0, metadata={"help": "The rank of the current process on the local node."})
    local_world_size: int = field(default=1, metadata={"help": "The total number of processes on the local node."})


def get_cpu_distributed_information() -> CPUInformation:
    """
    Returns various information about the environment in relation to CPU distributed training as a `CPUInformation`
    dataclass.
    """
    information = {}
    information["rank"] = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
    information["world_size"] = get_int_from_env(
        ["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1
    )
    information["local_rank"] = get_int_from_env(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
    )
    information["local_world_size"] = get_int_from_env(
        ["LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"],
        1,
    )
    return CPUInformation(**information)


def override_numa_affinity(local_process_index: int, verbose: Optional[bool] = None) -> None:
    """
    Overrides whatever NUMA affinity is set for the current process. This is very taxing and requires recalculating the
    affinity to set, ideally you should use `utils.environment.set_numa_affinity` instead.

    Args:
        local_process_index (int):
            The index of the current process on the current server.
        verbose (bool, *optional*):
            Whether to log out the assignment of each CPU. If `ACCELERATE_DEBUG_MODE` is enabled, will default to True.
    """
    if verbose is None:
        verbose = parse_flag_from_env("ACCELERATE_DEBUG_MODE", False)
    if torch.cuda.is_available():
        from accelerate.utils import is_pynvml_available

        if not is_pynvml_available():
            raise ImportError(
                "To set CPU affinity on CUDA GPUs the `pynvml` package must be available. (`pip install pynvml`)"
            )
        import pynvml as nvml

        # The below code is based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/BERT/gpu_affinity.py
        nvml.nvmlInit()
        num_elements = math.ceil(os.cpu_count() / 64)
        handle = nvml.nvmlDeviceGetHandleByIndex(local_process_index)
        affinity_string = ""
        for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = f"{j:064b}{affinity_string}"
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is the 0th element
        affinity_to_set = [i for i, e in enumerate(affinity_list) if e != 0]
        os.sched_setaffinity(0, affinity_to_set)
        if verbose:
            cpu_cores = os.sched_getaffinity(0)
            logger.info(f"Assigning {len(cpu_cores)} cpu cores to process {local_process_index}: {cpu_cores}")


@lru_cache
def set_numa_affinity(local_process_index: int, verbose: Optional[bool] = None) -> None:
    """
    Assigns the current process to a specific NUMA node. Ideally most efficient when having at least 2 cpus per node.

    This result is cached between calls. If you want to override it, please use
    `accelerate.utils.environment.override_numa_afifnity`.

    Args:
        local_process_index (int):
            The index of the current process on the current server.
        verbose (bool, *optional*):
            Whether to print the new cpu cores assignment for each process. If `ACCELERATE_DEBUG_MODE` is enabled, will
            default to True.
    """
    override_numa_affinity(local_process_index=local_process_index, verbose=verbose)


@contextmanager
def clear_environment():
    """
    A context manager that will temporarily clear environment variables.

    When this context exits, the previous environment variables will be back.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import clear_environment

    >>> os.environ["FOO"] = "bar"
    >>> with clear_environment():
    ...     print(os.environ)
    ...     os.environ["FOO"] = "new_bar"
    ...     print(os.environ["FOO"])
    {}
    new_bar

    >>> print(os.environ["FOO"])
    bar
    ```
    """
    _old_os_environ = os.environ.copy()
    os.environ.clear()

    try:
        yield
    finally:
        os.environ.clear()  # clear any added keys,
        os.environ.update(_old_os_environ)  # then restore previous environment


@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.

    Will convert the values in `kwargs` to strings and upper-case all the keys.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import patch_environment

    >>> with patch_environment(FOO="bar"):
    ...     print(os.environ["FOO"])  # prints "bar"
    >>> print(os.environ["FOO"])  # raises KeyError
    ```
    """
    existing_vars = {}
    for key, value in kwargs.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in kwargs:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)
