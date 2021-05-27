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

import importlib
import os
from distutils.util import strtobool
from enum import Enum

import torch


try:
    import torch_ccl  # noqa: F401

    _ccl_available = True
except ImportError:
    _ccl_available = False


try:
    import torch_xla.core.xla_model as xm

    _tpu_available = True
except ImportError:
    _tpu_available = False


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def is_ccl_available():
    return _ccl_available


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_tpu_available():
    return _tpu_available


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


def parse_flag_from_env(key, default=False):
    value = os.environ.get(key, str(default))
    return strtobool(value) == 1  # As its name indicates `strtobool` actually returns an int...


class DistributedType(str, Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    DEEPSPEED = "DEEPSPEED"
    TPU = "TPU"


class SageMakerDistributedType(str, Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **DATA_PARALLEL** -- using sagemaker distributed data parallelism.
        - **MODEL_PARALLEL** -- using sagemaker distributed model parallelism.
    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    DATA_PARALLEL = "DATA_PARALLEL"
    MODEL_PARALLEL = "MODEL_PARALLEL"


class ComputeEnvironment(str, Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """

    # Subclassing str as well as Enum allows the `ComputeEnvironment` to be JSON-serializable out of the box.
    LOCAL_MACHINE = "LOCAL_MACHINE"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"


# Inspired by Alex Martelli's 'Borg'.
class AcceleratorState:
    """
    This is a variation of a `singleton class <https://en.wikipedia.org/wiki/Singleton_pattern>`__ in the sense that
    all instance of :obj:`AcceleratorState` share the same state, which is initialized on the first instantiation.

    Attributes

        - **device** (:obj:`torch.device`) -- The device to use.
        - **distributed_type** (:obj:`~accelerate.state.DistributedType`) -- The type of distributed environment
          currently in use.
        - **num_processes** (:obj:`int`) -- The number of processes currently launched in parallel.
        - **process_index** (:obj:`int`) -- The index of the current process.
        - **local_process_index** (:obj:`int`) -- The index of the current process on the current server.
        - **use_fp16** (:obj:`bool`) -- Whether or not the current script will use mixed precision.
    """

    _shared_state = {}

    def __init__(self, fp16: bool = None, cpu: bool = False, deepspeed_plugin=None, _from_accelerator: bool = False):
        self.__dict__ = self._shared_state
        if not getattr(self, "initialized", False):
            self.backend = None
            self.deepspeed_plugin = None
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            elif is_tpu_available() and not cpu:
                self.distributed_type = DistributedType.TPU
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                self.local_process_index = xm.get_local_ordinal()
                self.device = xm.xla_device()
                self.use_fp16 = False
            elif os.environ.get("USE_DEEPSPEED", "false") == "true" and not cpu:
                assert (
                    is_deepspeed_available()
                ), "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                    self.backend = "nccl"
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self.use_fp16 = False  # deepspeed handles fp16 using deepspeed_config
                fp16 = parse_flag_from_env("USE_FP16", False) if fp16 is None else fp16
                deepspeed_plugin.deepspeed_config.update({"fp16": {"enabled": fp16}})
                self.deepspeed_plugin = deepspeed_plugin
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                self.distributed_type = DistributedType.MULTI_GPU
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                    self.backend = "nccl"
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self.use_fp16 = parse_flag_from_env("USE_FP16", False) if fp16 is None else fp16
            elif get_int_from_env(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"], 1) > 1:
                self.distributed_type = DistributedType.MULTI_CPU
                if is_ccl_available() and get_int_from_env(["CCL_WORKER_COUNT"], 0) > 0:
                    backend = "ccl"
                elif torch.distributed.is_mpi_available():
                    backend = "mpi"
                else:
                    backend = "gloo"
                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_rank = get_int_from_env(
                    ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
                )
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                self.local_process_index = local_rank
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size and backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend, rank=rank, world_size=size)
                    self.backend = backend
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = local_rank
                self.device = torch.device("cpu")
                self.use_fp16 = False
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
                self.use_fp16 = parse_flag_from_env("USE_FP16", False) if fp16 is None else fp16
            self.initialized = True

    def __repr__(self):
        use_fp16 = self.deepspeed_plugin.fp16 if self.distributed_type == DistributedType.DEEPSPEED else self.use_fp16
        repr = (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
            f"Use FP16 precision: {use_fp16}\n"
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            repr += f"ds_config: {self.deepspeed_plugin.ds_config}\n"
        return repr
