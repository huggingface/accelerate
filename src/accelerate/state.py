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
import warnings

import torch

from .utils import (
    DistributedType,
    DynamoBackend,
    get_ccl_version,
    get_int_from_env,
    is_ccl_available,
    is_deepspeed_available,
    is_tpu_available,
    parse_choice_from_env,
    parse_flag_from_env,
)
from .utils.dataclasses import SageMakerDistributedType


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def is_initialized() -> bool:
    """
    Checks if the `AcceleratorState` has been initialized from `Accelerator`. Same as `AcceleratorState.initialized`,
    but works as a module method.
    """
    return AcceleratorState._shared_state != {}


# Inspired by Alex Martelli's 'Borg'.
class AcceleratorState:
    """
    Singleton class that has information about the current training environment.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **initialized** (`bool`) -- Whether or not the `AcceleratorState` has been initialized from `Accelerator`.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
    """

    _shared_state = {}

    def __init__(
        self,
        mixed_precision: str = None,
        cpu: bool = False,
        dynamo_backend=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        megatron_lm_plugin=None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("ACCELERATE_USE_CPU"):
            cpu = True
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            self.backend = None
            self.deepspeed_plugin = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            dynamo_backend = (
                parse_choice_from_env("ACCELERATE_DYNAMO_BACKEND", "no") if dynamo_backend is None else dynamo_backend
            )
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None
            self.dynamo_backend = DynamoBackend(dynamo_backend.upper())
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            if (
                os.environ.get("ACCELERATE_USE_SAGEMAKER", "false") == "true"
                and os.environ.get("ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE") != SageMakerDistributedType.NO
                and not cpu
            ):
                if os.environ.get("ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE") == SageMakerDistributedType.DATA_PARALLEL:
                    self.distributed_type = DistributedType.MULTI_GPU
                    import smdistributed.dataparallel.torch.torch_smddp  # noqa

                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend="smddp")
                    self.backend = "smddp"
                    self.num_processes = torch.distributed.get_world_size()
                    self.process_index = torch.distributed.get_rank()
                    self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                    if self.device is None:
                        self.device = torch.device("cuda", self.local_process_index)
                    torch.cuda.set_device(self.device)
                    self._mixed_precision = mixed_precision
            elif is_tpu_available() and not cpu:
                self.distributed_type = DistributedType.TPU
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                self.local_process_index = xm.get_local_ordinal()
                self.device = xm.xla_device()
                if mixed_precision == "bf16":
                    if os.environ.get("ACCELERATE_DOWNCAST_BF16"):
                        os.environ["XLA_USE_BF16"] = str(0)
                        os.environ["XLA_DOWNCAST_BF16"] = str(1)
                        self.downcast_bfloat = True
                    else:
                        os.environ["XLA_USE_BF16"] = str(1)
                        os.environ["XLA_DOWNCAST_BF16"] = str(0)
                        self.downcast_bfloat = False
                self._mixed_precision = mixed_precision
            elif os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" and not cpu:
                assert (
                    is_deepspeed_available()
                ), "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    from .utils import compare_versions

                    self.backend = "nccl"
                    if compare_versions("deepspeed", ">", "0.6.5"):
                        from deepspeed import comm as dist

                        dist.init_distributed(dist_backend=self.backend)
                    else:
                        torch.distributed.init_process_group(backend="nccl", **kwargs)

                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                if self.device is None:
                    self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self._mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
                self.deepspeed_plugin = deepspeed_plugin
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                self.distributed_type = DistributedType.MULTI_GPU
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl", **kwargs)
                    self.backend = "nccl"
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                if self.device is None:
                    self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self._mixed_precision = mixed_precision
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != "no":
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
                if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true":
                    self.distributed_type = DistributedType.MEGATRON_LM
                    megatron_lm_plugin.set_mixed_precision(self._mixed_precision)
                    self.megatron_lm_plugin = megatron_lm_plugin
            elif get_int_from_env(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"], 1) > 1:
                self.distributed_type = DistributedType.MULTI_CPU
                if is_ccl_available() and get_int_from_env(["CCL_WORKER_COUNT"], 0) > 0:
                    if get_ccl_version() >= "1.12":
                        import oneccl_bindings_for_pytorch  # noqa: F401
                    else:
                        import torch_ccl  # noqa: F401
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
                if self.device is None:
                    self.device = torch.device("cpu")
                self._mixed_precision = mixed_precision
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                if parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE") and not cpu:
                    if not torch.backends.mps.is_available():
                        if not torch.backends.mps.is_built():
                            raise AssertionError(
                                "MPS not available because the current PyTorch install was not "
                                "built with MPS enabled. Please install torch version >=1.12.0 on "
                                "your Apple silicon Mac running macOS 12.3 or later with a native "
                                "version (arm64) of Python"
                            )
                        else:
                            raise AssertionError(
                                "MPS not available because the current MacOS version is not 12.3+ "
                                "and/or you do not have an MPS-enabled device on this machine."
                            )
                    else:
                        from .utils import is_torch_version

                        if not is_torch_version(">", "1.12.0"):
                            warnings.warn(
                                "We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing) on your MacOS machine. "
                                "It has major fixes related to model correctness and performance improvements for transformer based models. "
                                "Please refer to https://github.com/pytorch/pytorch/issues/82707 for more details."
                            )
                        if self.device is None:
                            self.device = torch.device("mps")
                elif self.device is None:
                    if cpu or not torch.cuda.is_available():
                        self.device = torch.device("cpu")
                    else:
                        self.device = torch.device("cuda")
                self._mixed_precision = mixed_precision

            if (
                self.dynamo_backend != DynamoBackend.NO
                and self._mixed_precision == "no"
                and self.device.type == "cuda"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True

        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)

    def __repr__(self):
        repr = (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
            f"Mixed precision type: {self.mixed_precision}\n"
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            repr += f"ds_config: {self.deepspeed_plugin.deepspeed_config}\n"
        return repr

    # For backward compatibility
    @property
    def use_fp16(self):
        return self._mixed_precision != "no"

    @property
    def mixed_precision(self):
        if self.distributed_type == DistributedType.DEEPSPEED:
            config = self.deepspeed_plugin.deepspeed_config
            if config.get("fp16", {}).get("enabled", False):
                mixed_precision = "fp16"
            elif config.get("bf16", {}).get("enabled", False):
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"
        else:
            mixed_precision = self._mixed_precision
        return mixed_precision

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        AcceleratorState._shared_state = {}

    @property
    def initialized(self) -> bool:
        "Returns whether the `AcceleratorState` has been initialized"
        return self._shared_state != {}

    def _check_initialized(self, mixed_precision=None, cpu=None):
        "Checks if a modification is trying to be made and the `AcceleratorState` has already been initialized"
        if self.initialized:
            err = "AcceleratorState has already been initialized and cannot be changed, restart your runtime completely and pass `{flag}` to `Accelerate()`."
            if cpu and self.device.type != "cpu":
                raise ValueError(err.format(flag="cpu=True"))
            if mixed_precision is not None and mixed_precision != self._mixed_precision:
                raise ValueError(err.format(flag=f"mixed_precision='{mixed_precision}'"))


class GradientState:
    """
    Singleton class that has information related to gradient synchronization for gradient accumulation

    **Available attributes:**

        - **end_of_dataloader** (`bool`) -- Whether we have reached the end the current dataloader
        - **remainder** (`int`) -- The number of extra samples that were added from padding the dataloader
        - **sync_gradients** (`bool`) -- Whether the gradients should be synced across all devices
    """

    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self.end_of_dataloader = False
            self.remainder = -1

    @property
    def initialized(self) -> bool:
        "Returns whether the `GradientState` has been initialized"
        return GradientState._shared_state != {}

    def __repr__(self):
        return (
            f"Sync Gradients: {self.sync_gradients}\n"
            f"At end of current dataloader: {self.end_of_dataloader}\n"
            f"Extra samples added: {self.remainder}"
        )

    def _set_sync_gradients(self, sync_gradients):
        "Private function that sets whether gradients should be synchronized. Users should not have to call this."
        self.sync_gradients = sync_gradients

    def _set_end_of_dataloader(self, end_of_dataloader):
        "Private function that sets whether the end of the current dataloader has been reached. Users should not have to call this."
        self.end_of_dataloader = end_of_dataloader

    def _set_remainder(self, remainder):
        "Private function that sets the number of remaining samples at the end of the dataloader. Users should not have to call this."
        self.remainder = remainder
