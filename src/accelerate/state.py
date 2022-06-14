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

import os
from copy import deepcopy
from distutils.util import strtobool

import torch

from .utils import DistributedType, is_ccl_available, is_deepspeed_available, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def parse_flag_from_env(key, default=False):
    value = os.environ.get(key, str(default))
    return strtobool(value) == 1  # As its name indicates `strtobool` actually returns an int...


def parse_choice_from_env(key, default="no"):
    value = os.environ.get(key, str(default))
    return value


# Inspired by Alex Martelli's 'Borg'.
class AcceleratorState:
    """
    This is a variation of a [singleton class](https://en.wikipedia.org/wiki/Singleton_pattern) in the sense that all
    instance of `AcceleratorState` share the same state, which is initialized on the first instantiation.

    Attributes:

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** (`~accelerate.state.DistributedType`) -- The type of distributed environment currently
          in use.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision. If you are using
          mixed precision, define if you want to use FP16 or BF16 (bfloat16) as the floating point.
    """

    _shared_state = {}

    def __init__(
        self,
        mixed_precision: str = None,
        cpu: bool = False,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("USE_CPU"):
            cpu = True
        if not getattr(self, "initialized", False):
            self.backend = None
            self.deepspeed_plugin = None
            mixed_precision = mixed_precision.lower() if mixed_precision else None
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
                self.mixed_precision = "no"
            elif os.environ.get("USE_DEEPSPEED", "false") == "true" and not cpu:
                assert (
                    is_deepspeed_available()
                ), "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl", **kwargs)
                    self.backend = "nccl"
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self.mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
                self.deepspeed_plugin = deepspeed_plugin
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                self.distributed_type = DistributedType.MULTI_GPU
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl", **kwargs)
                    self.backend = "nccl"
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self.mixed_precision = (
                    parse_choice_from_env("MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
                )
                if os.environ.get("USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if self.mixed_precision != "no":
                        raise ValueError(
                            "Mixed precision is currently not supported for FSDP. Please set `mixed_precision` to `no`."
                        )
                    self.fsdp_plugin = fsdp_plugin
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
                    torch.distributed.init_process_group(backend, rank=rank, world_size=size, **kwargs)
                    self.backend = backend
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = local_rank
                self.device = torch.device("cpu")
                self.mixed_precision = "no"
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
                self.mixed_precision = (
                    parse_choice_from_env("MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
                )
            self.initialized = True

    def __repr__(self):
        mixed_precision = self.mixed_precision

        repr = (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            repr += f"ds_config: {self.deepspeed_plugin.deepspeed_config}\n"
        else:
            f"Mixed precision type: {mixed_precision}\n"
        return repr

    # For backward compatibility
    @property
    def use_fp16(self):
        return self.mixed_precision != "no"

    @staticmethod
    def is_deepspeed_zero3_enabled():
        if (
            "deepspeed_plugin" in AcceleratorState._shared_state
            and AcceleratorState._shared_state["deepspeed_plugin"] is not None
        ):
            deepspeed_plugin = AcceleratorState._shared_state["deepspeed_plugin"]
            return deepspeed_plugin.hf_ds_config.is_zero3() and deepspeed_plugin.zero3_init_flag
        return False

    @staticmethod
    def get_deepspeed_config():
        ds_config = None
        if (
            "deepspeed_plugin" in AcceleratorState._shared_state
            and AcceleratorState._shared_state["deepspeed_plugin"] is not None
        ):
            ds_config = deepcopy(AcceleratorState._shared_state["deepspeed_plugin"].hf_ds_config.config)
            if ds_config["gradient_accumulation_steps"] == "auto":
                ds_config["gradient_accumulation_steps"] = 1
            if (
                "train_micro_batch_size_per_gpu" not in ds_config
                or ds_config["train_micro_batch_size_per_gpu"] == "auto"
            ):
                ds_config["train_micro_batch_size_per_gpu"] = 1
            if ds_config["train_batch_size"] == "auto":
                del ds_config["train_batch_size"]
        return ds_config

    @staticmethod
    def set_deepspeed_config(hf_ds_config):
        from .utils import DeepSpeedPlugin, HfDeepSpeedConfig

        if isinstance(hf_ds_config, HfDeepSpeedConfig):
            AcceleratorState._shared_state["deepspeed_plugin"] = DeepSpeedPlugin(
                hf_ds_config=hf_ds_config, zero3_init_flag=True
            )
        else:
            raise ValueError("`hf_ds_config` must be an instance of `accelerate.utils.HfDeepSpeedConfig`")

    @staticmethod
    def unset_deepspeed_config():
        AcceleratorState._shared_state["deepspeed_plugin"] = None
