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
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional

import torch

from .utils import (
    DistributedType,
    DynamoBackend,
    GradientAccumulationPlugin,
    get_ccl_version,
    get_int_from_env,
    is_ccl_available,
    is_deepspeed_available,
    is_mps_available,
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


# Lambda function that does nothing
def do_nothing(*args, **kwargs):
    return None


# Inspired by Alex Martelli's 'Borg'.
class PartialState:
    """
    Singleton class that has information about the current training environment and functions to help with process
    control. Designed to be used when only process control and device execution states are needed. Does *not* need to
    be initialized from `Accelerator`.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
    """

    _shared_state = {}

    def __init__(self, cpu: bool = False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None
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
            elif is_tpu_available() and not cpu:
                self.distributed_type = DistributedType.TPU
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                self.local_process_index = xm.get_local_ordinal()
                self.device = xm.xla_device()
            elif os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" and not cpu:
                assert (
                    is_deepspeed_available()
                ), "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl", **kwargs)

                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                if self.device is None:
                    self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
                self._mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
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
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0

                # the below block using env variable for `mps` will be removed in version 0.18.0
                if parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE") and not cpu:
                    from .utils import is_torch_version

                    if is_mps_available():
                        if not is_torch_version(">", "1.12.0"):
                            warnings.warn(
                                "We strongly recommend to install PyTorch >= 1.13 for transformer based models."
                            )
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        self.device = torch.device("mps")
                    else:
                        raise AssertionError(
                            "MPS not available because PyTorch version is < 1.12.0 or MacOS version is < 12.3 "
                            "and/or you do not have an MPS-enabled device on this machine."
                        )

                if self.device is None:
                    if cpu or not (torch.cuda.is_available() or is_mps_available()):
                        self.device = torch.device("cpu")
                    elif is_mps_available():
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        self.device = torch.device("mps")
                    else:
                        self.device = torch.device("cuda")
        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)

    def __repr__(self) -> str:
        return (
            f"Distributed environment: {self.distributed_type}{('  Backend: ' + self.backend) if self.backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
        )

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        PartialState._shared_state = {}

    @property
    def initialized(self) -> bool:
        "Returns whether the `PartialState` has been initialized"
        return self._shared_state != {}

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.distributed_type != DistributedType.NO and self.num_processes > 1

    @property
    def is_last_process(self) -> bool:
        "Returns whether the current process is the last one"
        return self.process_index == self.num_processes - 1

    @property
    def is_main_process(self) -> bool:
        "Returns whether the current process is the main process"
        return (
            self.process_index == 0 if self.distributed_type != DistributedType.MEGATRON_LM else self.is_last_process
        )

    @property
    def is_local_main_process(self) -> bool:
        "Returns whether the current process is the main process on the local node"
        return (
            self.local_process_index == 0
            if self.distributed_type != DistributedType.MEGATRON_LM
            else self.is_last_process
        )

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> if state.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> state.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        if self.distributed_type in (
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_CPU,
            DistributedType.DEEPSPEED,
            DistributedType.FSDP,
        ):
            torch.distributed.barrier()
        elif self.distributed_type == DistributedType.TPU:
            xm.rendezvous("accelerate.utils.wait_for_everyone")

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        yield from self._goes_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> with state.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {state.local_process_index}")
        ```
        """
        yield from self._goes_first(self.is_local_main_process)

    def on_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()


        >>> @state.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        if not self.initialized:
            raise ValueError("The `PartialState` or `Accelerator` must be initialized before calling this function.")
        if self.is_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_local_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the local main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        if self.is_local_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_last_process(self, function: Callable[..., Any]):
        """
        Decorator that only runs the decorated function on the last process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_last_process
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        if self.is_last_process or not self.use_distributed:
            return function
        return do_nothing

    def on_process(self, function: Callable[..., Any] = None, process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_process, process_index=process_index)
        if (self.process_index == process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def on_local_process(self, function: Callable[..., Any] = None, local_process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index on the current node.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_local_process, local_process_index=local_process_index)
        if (self.local_process_index == local_process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def print(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)


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
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
    """

    _shared_state = {}

    def __init__(
        self,
        mixed_precision: str = None,
        cpu: bool = False,
        dynamo_plugin=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        megatron_lm_plugin=None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("ACCELERATE_USE_CPU"):
            cpu = True
        if PartialState._shared_state == {}:
            PartialState(cpu, **kwargs)
        self.__dict__.update(PartialState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            self.deepspeed_plugin = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            self.dynamo_plugin = dynamo_plugin
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            # deepspeed handles mixed_precision using deepspeed_config
            self._mixed_precision = "no" if self.distributed_type == DistributedType.DEEPSPEED else mixed_precision
            if self.distributed_type == DistributedType.TPU:
                if mixed_precision == "bf16":
                    if os.environ.get("ACCELERATE_DOWNCAST_BF16"):
                        os.environ["XLA_USE_BF16"] = str(0)
                        os.environ["XLA_DOWNCAST_BF16"] = str(1)
                        self.downcast_bfloat = True
                    else:
                        os.environ["XLA_USE_BF16"] = str(1)
                        os.environ["XLA_DOWNCAST_BF16"] = str(0)
                        self.downcast_bfloat = False
            elif os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" and not cpu:
                self.deepspeed_plugin = deepspeed_plugin
            elif self.distributed_type == DistributedType.MULTI_GPU:
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != "no":
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
                if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true":
                    self.distributed_type = DistributedType.MEGATRON_LM
                    megatron_lm_plugin.set_mixed_precision(self._mixed_precision)
                    self.megatron_lm_plugin = megatron_lm_plugin
            if (
                self.dynamo_plugin.backend != DynamoBackend.NO
                and self._mixed_precision == "no"
                and self.device.type == "cuda"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
            PartialState._shared_state["distributed_type"] = self.distributed_type

    @property
    def initialized(self) -> bool:
        return self._shared_state != PartialState._shared_state

    def __repr__(self):
        repr = PartialState().__repr__() + f"\nMixed precision type: {self.mixed_precision}\n"
        if self.distributed_type == DistributedType.DEEPSPEED:
            repr += f"ds_config: {self.deepspeed_plugin.deepspeed_config}\n"
        return repr

    def _check_initialized(self, mixed_precision=None, cpu=None):
        "Checks if a modification is trying to be made and the `AcceleratorState` has already been initialized"
        if self.initialized:
            err = "AcceleratorState has already been initialized and cannot be changed, restart your runtime completely and pass `{flag}` to `Accelerator()`."
            if cpu and self.device.type != "cpu":
                raise ValueError(err.format(flag="cpu=True"))
            if (
                mixed_precision is not None
                and mixed_precision != self._mixed_precision
                and self.distributed_type != DistributedType.DEEPSPEED
            ):
                raise ValueError(err.format(flag=f"mixed_precision='{mixed_precision}'"))

    # For backward compatibility
    @property
    def use_fp16(self):
        warnings.warn(
            "The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use "
            "`AcceleratorState.mixed_precision == 'fp16'` instead.",
            FutureWarning,
        )
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
    def _reset_state(reset_partial_state: bool = False):
        "Resets `_shared_state`, is used internally and should not be called"
        AcceleratorState._shared_state = {}
        if reset_partial_state:
            PartialState._reset_state()

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return PartialState().use_distributed

    @property
    def is_last_process(self) -> bool:
        "Returns whether the current process is the last one"
        return PartialState().is_last_process

    @property
    def is_main_process(self) -> bool:
        "Returns whether the current process is the main process"
        return PartialState().is_main_process

    @property
    def is_local_main_process(self) -> bool:
        "Returns whether the current process is the main process on the local node"
        return PartialState().is_local_main_process

    def wait_for_everyone(self):
        PartialState().wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield PartialState().main_process_first()

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield PartialState().local_main_process_first()

    def print(self, *args, **kwargs):
        PartialState().print(*args, **kwargs)


class GradientState:
    """
    Singleton class that has information related to gradient synchronization for gradient accumulation

    **Available attributes:**

        - **end_of_dataloader** (`bool`) -- Whether we have reached the end the current dataloader
        - **remainder** (`int`) -- The number of extra samples that were added from padding the dataloader
        - **sync_gradients** (`bool`) -- Whether the gradients should be synced across all devices
        - **active_dataloader** (`Optional[DataLoader]`) -- The dataloader that is currently being iterated over
        - **dataloader_references** (`List[Optional[DataLoader]]`) -- A list of references to the dataloaders that are
          being iterated over
        - **num_steps** (`int`) -- The number of steps to accumulate over
        - **adjust_scheduler** (`bool`) -- Whether the scheduler should be adjusted to account for the gradient
          accumulation
    """

    _shared_state = {}

    def __init__(self, gradient_accumulation_plugin: Optional[GradientAccumulationPlugin] = None):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self.end_of_dataloader = False
            self.remainder = -1
            self.active_dataloader = None
            self.dataloader_references = [None]
            self.plugin_kwargs = gradient_accumulation_plugin.to_kwargs()

        # Plugin args are different and can be updated
        if gradient_accumulation_plugin is not None and self.plugin_kwargs != gradient_accumulation_plugin.to_kwargs():
            self.plugin_kwargs = gradient_accumulation_plugin.to_kwargs()

    @property
    def num_steps(self) -> int:
        "Returns the number of steps to accumulate over"
        return self.plugin_kwargs.get("num_steps", 1)

    @property
    def adjust_scheduler(self) -> bool:
        "Returns whether the scheduler should be adjusted"
        return self.plugin_kwargs.get("adjust_scheduler", False)

    @property
    def initialized(self) -> bool:
        "Returns whether the `GradientState` has been initialized"
        return GradientState._shared_state != {}

    def __repr__(self):
        return (
            f"Sync Gradients: {self.sync_gradients}\n"
            f"At end of current dataloader: {self.end_of_dataloader}\n"
            f"Extra samples added: {self.remainder}\n"
            f"Gradient accumulation plugin: {self.plugin_kwargs}\n"
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

    def _add_dataloader(self, dataloader):
        "Private function that adds a dataloader to `self.dataloader_references` and sets `in_dataloader` to `True`. Users should not have to call this."
        self.active_dataloader = dataloader
        self.dataloader_references.append(self.active_dataloader)
        self._set_end_of_dataloader(False)

    def _remove_dataloader(self, dataloader):
        "Private function that removes a dataloader from `self.dataloader_references` and sets `in_dataloader` to `False` if there are no more dataloaders. Users should not have to call this."
        self.dataloader_references.remove(dataloader)
        self.active_dataloader = self.dataloader_references[-1]
        self._set_end_of_dataloader(True)

    @property
    def in_dataloader(self) -> bool:
        "Returns whether the current process is in a dataloader"
        return self.active_dataloader is not None

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        GradientState._shared_state = {}
