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
from functools import lru_cache, wraps
from shutil import which
from typing import Optional, Union

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


def str_to_bool(value, to_bool: bool = False) -> Union[int, bool]:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1 if not to_bool else True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0 if not to_bool else False
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


def are_libraries_initialized(*library_names: str) -> list[str]:
    """
    Checks if any of `library_names` are imported in the environment. Will return any names that are.
    """
    return [lib_name for lib_name in library_names if lib_name in sys.modules.keys()]


def get_current_device_type() -> tuple[str, str]:
    """Return the launcher device and distributed type without initializing an accelerator."""
    from .imports import get_distributed_backend, get_torch_accelerator_backend

    device_type, _ = get_torch_accelerator_backend(check_available=False)
    if device_type is None:
        return "cpu", "NO"
    if get_distributed_backend(device_type) is None:
        return device_type, "NO"
    if device_type == "cuda":
        return device_type, "MULTI_GPU"
    return device_type, f"MULTI_{device_type.upper()}"


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

    Notably uses `nvidia-smi` instead of torch to not initialize CUDA.
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


@lru_cache
def check_cuda_fp8_capability():
    """
    Checks if the current GPU available supports FP8.

    Notably might initialize `torch.cuda` to check.
    """

    try:
        # try to get the compute capability from nvidia-smi
        output = subprocess.check_output(
            [_nvidia_smi(), "--query-gpu=compute_capability", "--format=csv,noheader"], universal_newlines=True
        )
        output = output.strip()
        # we take the first GPU's compute capability
        compute_capability = tuple(map(int, output.split(os.linesep)[0].split(".")))
    except Exception:
        compute_capability = torch.cuda.get_device_capability()

    return compute_capability >= (8, 9)


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


def is_slurm_available() -> bool:
    """
    Detects whether the current environment is a SLURM (Simple Linux Utility for Resource Management)
    scheduled job environment.

    Checks for the presence of key SLURM environment variables that are set by `sbatch`/`srun`
    when a job is submitted to a SLURM cluster.

    Returns:
        `True` if running inside a SLURM job, `False` otherwise.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import is_slurm_available
    >>> os.environ["SLURM_JOB_ID"] = "12345"
    >>> is_slurm_available()
    True
    >>> del os.environ["SLURM_JOB_ID"]
    >>> is_slurm_available()
    False
    ```
    """
    # SLURM sets SLURM_JOB_ID when inside a scheduled job.
    # SLURM_PROCID and SLURM_NNODES are also commonly present in multi-node jobs.
    slurm_indicators = ["SLURM_JOB_ID", "SLURM_PROCID", "SLURM_NNODES", "SLURM_NODEID"]
    return any(os.environ.get(key) is not None for key in slurm_indicators)


@dataclass
class SlurmInformation:
    """
    Stores SLURM-specific information derived from environment variables.

    Attributes:
        job_id: The SLURM job ID (`SLURM_JOB_ID`).
        num_nodes: Total number of nodes allocated to this job (`SLURM_NNODES`).
        node_id: The rank of the current node (0-indexed, `SLURM_NODEID`).
        num_tasks: Total number of tasks in the job (`SLURM_NTASKS`).
        tasks_per_node: Number of tasks per node (`SLURM_NTASKS_PER_NODE`).
        procid: Global process rank within SLURM (`SLURM_PROCID`).
        local_gpus_per_node: Number of GPUs per node, derived from `SLURM_GPUS_PER_NODE`
            or `SLURM_GPUS_ON_NODE` if available. May be `None` if not set.
        node_list: The raw node list string (`SLURM_JOB_NODELIST`).
        nodelist_filename: The path to the file containing expanded node list (`SLURM_NODELIST`).
    """

    job_id: Optional[str] = None
    num_nodes: int = 1
    node_id: int = 0
    num_tasks: Optional[int] = None
    tasks_per_node: Optional[int] = None
    procid: Optional[int] = None
    local_gpus_per_node: Optional[int] = None
    node_list: Optional[str] = None
    nodelist_filename: Optional[str] = None

    @property
    def world_size(self) -> int:
        """
        The total number of processes for distributed training.

        Derived from `SLURM_NTASKS` (if set), or `num_nodes * gpus_per_node`.
        Falls back to `1` when no SLURM variables are present.
        """
        if self.num_tasks is not None:
            return self.num_tasks
        if self.local_gpus_per_node is not None:
            return self.num_nodes * self.local_gpus_per_node
        return 1


def get_slurm_distributed_information() -> SlurmInformation:
    """
    Extracts distributed-training-relevant information from SLURM environment variables.

    When running under `srun`, SLURM sets several environment variables that describe the
    job layout. This function parses them into a `SlurmInformation` dataclass so that
    callers (e.g. `accelerate launch`) can auto-configure themselves.

    Recognized SLURM variables:
        - `SLURM_JOB_ID` -- job identifier
        - `SLURM_NNODES` -- total nodes allocated
        - `SLURM_NODEID` -- rank of this node (0-indexed)
        - `SLURM_NTASKS` -- total task count across the whole job
        - `SLURM_NTASKS_PER_NODE` -- tasks per node
        - `SLURM_PROCID` -- global rank of the current SLURM task
        - `SLURM_GPUS_PER_NODE` / `SLURM_GPUS_ON_NODE` -- GPUs allocated to each node
        - `SLURM_JOB_NODELIST` / `SLURM_NODELIST` -- node allocation list

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import get_slurm_distributed_information, is_slurm_available
    >>> os.environ.update({
    ...     "SLURM_JOB_ID": "12345",
    ...     "SLURM_NNODES": "2",
    ...     "SLURM_NODEID": "0",
    ...     "SLURM_NTASKS": "16",
    ...     "SLURM_PROCID": "0",
    ...     "SLURM_GPUS_PER_NODE": "8",
    ... })
    >>> info = get_slurm_distributed_information()
    >>> info.num_nodes
    2
    >>> info.node_id
    0
    >>> info.world_size
    16
    ```
    """
    # Helper: return int from env var or None
    def _int_or_none(key: str) -> Optional[int]:
        val = os.environ.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    job_id = os.environ.get("SLURM_JOB_ID")
    num_nodes = _int_or_none("SLURM_NNODES") or 1
    node_id = _int_or_none("SLURM_NODEID") or 0
    num_tasks = _int_or_none("SLURM_NTASKS")
    tasks_per_node = _int_or_none("SLURM_NTASKS_PER_NODE")
    procid = _int_or_none("SLURM_PROCID")
    # GPU count: several SLURM environments expose this differently.
    local_gpus_per_node = _int_or_none("SLURM_GPUS_PER_NODE")
    if local_gpus_per_node is None:
        local_gpus_per_node = _int_or_none("SLURM_GPUS_ON_NODE")
    # Fall back to counting devices if torch is available
    if local_gpus_per_node is None:
        try:
            import torch

            if torch.cuda.is_available():
                local_gpus_per_node = torch.cuda.device_count()
        except Exception:
            local_gpus_per_node = None

    node_list = os.environ.get("SLURM_JOB_NODELIST")
    nodelist_filename = os.environ.get("SLURM_NODELIST")

    return SlurmInformation(
        job_id=job_id,
        num_nodes=num_nodes,
        node_id=node_id,
        num_tasks=num_tasks,
        tasks_per_node=tasks_per_node,
        procid=procid,
        local_gpus_per_node=local_gpus_per_node,
        node_list=node_list,
        nodelist_filename=nodelist_filename,
    )


def get_slurm_master_addr() -> Optional[str]:
    """
    Determines the master node IP/hostname for a SLURM multi-node job.

    Strategy (tried in order):
        1. If `MASTER_ADDR` is already set in the environment, return it.
        2. Otherwise, if we have a `SLURM_JOB_NODELIST` or `SLURM_NODELIST`, use
           the hostname of the first node (requires the external `scontrol`
           or `srun` binary to be available).
        3. If all else fails, return `None` and let the caller decide.

    Returns:
        A string hostname/IP, or `None` if it cannot be determined.
    """
    # 1. Already set?
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        return master_addr

    # 2. Try `scontrol show hostnames` to expand the SLURM node list and pick the first one
    node_list_env = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if node_list_env:
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", node_list_env],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # scontrol not available or timed out; fall through
            pass

    return None


def _parse_cpu_list(cpu_list_str: str) -> list[int]:
    """Parse a Linux sysfs-style CPU list (e.g. "0-7,16-23") into a list of ints."""
    cpus = []
    for part in cpu_list_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


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
        from accelerate.utils import is_amdsmi_available, is_pynvml_available, is_rocm_available

        affinity_to_set = None

        if is_rocm_available():
            if not is_amdsmi_available():
                raise ImportError(
                    "To set CPU affinity on ROCm GPUs the `amdsmi` package must be available. "
                    "It ships with ROCm; ensure the ROCm Python bindings are on PYTHONPATH."
                )
            import amdsmi

            amdsmi.amdsmi_init()
            try:
                handles = amdsmi.amdsmi_get_processor_handles()
                handle = handles[local_process_index]
                numa_node = amdsmi.amdsmi_topo_get_numa_node_number(handle)
                if numa_node is None or numa_node < 0:
                    # GPU is not bound to a NUMA node; fall back to all online CPUs
                    affinity_to_set = list(os.sched_getaffinity(0))
                else:
                    with open(f"/sys/devices/system/node/node{numa_node}/cpulist") as f:
                        cpu_list_str = f.read().strip()
                    affinity_to_set = _parse_cpu_list(cpu_list_str)
            finally:
                amdsmi.amdsmi_shut_down()
        else:
            if not is_pynvml_available():
                raise ImportError(
                    "To set CPU affinity on CUDA GPUs the `nvidia-ml-py` package must be available. (`pip install nvidia-ml-py`)"
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


def purge_accelerate_environment(func_or_cls):
    """Decorator to clean up accelerate environment variables set by the decorated class or function.

    In some circumstances, calling certain classes or functions can result in accelerate env vars being set and not
    being cleaned up afterwards. As an example, when calling:

    TrainingArguments(fp16=True, ...)

    The following env var will be set:

    ACCELERATE_MIXED_PRECISION=fp16

    This can affect subsequent code, since the env var takes precedence over TrainingArguments(fp16=False). This is
    especially relevant for unit testing, where we want to avoid the individual tests to have side effects on one
    another. Decorate the unit test function or whole class with this decorator to ensure that after each test, the env
    vars are cleaned up. This works for both unittest.TestCase and normal classes (pytest); it also works when
    decorating the parent class.

    """
    prefix = "ACCELERATE_"

    @contextmanager
    def env_var_context():
        # Store existing accelerate env vars
        existing_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        try:
            yield
        finally:
            # Restore original env vars or remove new ones
            for key in [k for k in os.environ if k.startswith(prefix)]:
                if key in existing_vars:
                    os.environ[key] = existing_vars[key]
                else:
                    os.environ.pop(key, None)

    def wrap_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with env_var_context():
                return func(*args, **kwargs)

        wrapper._accelerate_is_purged_environment_wrapped = True
        return wrapper

    if not isinstance(func_or_cls, type):
        return wrap_function(func_or_cls)

    # Handle classes by wrapping test methods
    def wrap_test_methods(test_class_instance):
        for name in dir(test_class_instance):
            if name.startswith("test"):
                method = getattr(test_class_instance, name)
                if callable(method) and not hasattr(method, "_accelerate_is_purged_environment_wrapped"):
                    setattr(test_class_instance, name, wrap_function(method))
        return test_class_instance

    # Handle inheritance
    wrap_test_methods(func_or_cls)
    func_or_cls.__init_subclass__ = classmethod(lambda cls, **kw: wrap_test_methods(cls))
    return func_or_cls
