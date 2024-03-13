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
import platform
import subprocess
import sys
from shutil import which
from typing import List

import torch


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


def _get_nvidia_smi():
    """
    Returns the right nvidia-smi command based on the system.
    """
    if platform.system() == "Windows":
        # If platform is Windows and nvidia-smi can't be found in path
        # try from systemd drive with default installation path
        command = which("nvidia-smi")
        if command is None:
            command = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ["systemdrive"]
    else:
        command = "nvidia-smi"
    return command


def get_gpu_info():
    """
    Gets GPU count and names using `nvidia-smi` instead of torch to not initialize CUDA.

    Largely based on the `gputil` library.
    """
    command = _get_nvidia_smi()
    # Returns as list of `n` GPUs and their names
    output = subprocess.check_output(
        [command, "--query-gpu=count,name", "--format=csv,noheader"], universal_newlines=True
    )
    output = output.strip()
    gpus = output.split(os.linesep)
    # Get names from output
    gpu_count = len(gpus)
    gpu_names = [gpu.split(",")[1].strip() for gpu in gpus]
    return gpu_names, gpu_count


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
                return False
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


def get_query_command(device_id: int = None, backend: str = None):
    """
    Returns the right param args for `numa` support based on a `backend`.

    Currently only supports `nvidia`.
    """
    if backend == "nvidia":
        return [_get_nvidia_smi(), "--query-gpu=pci.bus_id", "-i", str(device_id), "--format=csv,noheader"]
    else:
        # TODO: Add support for AMD and Gaudi2
        raise ValueError(f"Backend {backend} is currently not supported through `accelerate launch`.")


def get_bus_id(device_id: int = None, backend: str = None):
    """
    Gets the bus ID for `device_id`. If not passed, will default to the local rank.
    """
    if device_id is None:
        device_id = os.environ.get("LOCAL_RANK", 0)
    command = get_query_command(device_id, backend)
    output = subprocess.check_output(command, universal_newlines=True)
    # The output is a string like "00000000:00:04.0"
    # We want to return "0000:00:04.0"
    # As it can be passed to `ls -l /sys/bus/pci/devices/${bus_id}`
    return output.strip()[4:]


def get_numa_node_for_device(device_id: int = None, backend: str = None):
    """
    Gets the numa node for `device_id`. If not passed, will default to the local rank.
    """
    bus_id = get_bus_id(device_id, backend)
    if bus_id is None:
        raise ValueError(f"Expected to find a bus_id for device at {device_id}, but none found!")
    with open(f"/sys/bus/pci/devices/{bus_id}/numa_node") as f:
        return f.read().strip()
