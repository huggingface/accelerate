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

from __future__ import annotations

import logging
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional

import torch

from .utils import (
    DistributedType,
    DynamoBackend,
    GradientAccumulationPlugin,
    check_cuda_p2p_ib_support,
    check_fp8_capability,
    get_ccl_version,
    get_cpu_distributed_information,
    get_int_from_env,
    is_ccl_available,
    is_datasets_available,
    is_deepspeed_available,
    is_fp8_available,
    is_ipex_available,
    is_mlu_available,
    is_mps_available,
    is_npu_available,
    is_torch_xla_available,
    is_xpu_available,
    parse_choice_from_env,
    parse_flag_from_env,
    set_numa_affinity,
)
from .utils.dataclasses import SageMakerDistributedType


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_mlu_available(check_device=False):
    import torch_mlu  # noqa: F401

if is_npu_available(check_device=False):
    import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)


def is_initialized() -> bool:
    """
    Checks if the `AcceleratorState` has been initialized from `Accelerator`. Same as `AcceleratorState.initialized`,
    but works as a module method.
    """
    return AcceleratorState._shared_state != {}


# Lambda function that does nothing
def do_nothing(*args, **kwargs):
    return None


class ThreadLocalSharedDict(threading.local):
    """
    Descriptor that holds a dict shared between instances of a class in the same thread.

    Note: Descriptors have slightly different semantics than just a dict field on its own.
    `PartialState(...)._shared_state` and `PartialState._shared_state` (instance vs class) give the same value: the
    underlying _storage dict. Likewise, `PartialState(...)._shared_state = {...}` overrides the _storage dict inside
    the descriptor as you would expect. However, `PartialState._shared_state = {}` actually replaces the descriptor
    object with a dict instead Thus, you should modify the _storage dict in-place (e.g. `_shared_state.clear()`).

    See Python documentation for an explanation of descriptors: https://docs.python.org/3/howto/descriptor.html

    This is required for using PyTorch/XLA with PJRT in multithreaded mode (required for TPU v2 and v3).

    See https://github.com/pytorch/xla/blob/r2.0/docs/pjrt.md#multithreading-on-tpu-v2v3
    """

    def __init__(self, thread_local: bool = False):
        self._storage = {}

    def __get__(self, obj, objtype=None):
        return self._storage

    def __set__(self, obj, value):
        self._storage = value


# Prefer global shared dictionary, except when using TPU.
SharedDict = dict if not is_torch_xla_available() else ThreadLocalSharedDict


# Inspired by Alex Martelli's 'Borg'.
class PartialState:
    """
    Singleton class that has information about the current training environment and functions to help with process
    control. Designed to be used when only process control and device execution states are needed. Does *not* need to
    be initialized from `Accelerator`.

    Args:
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore any accelerators available if set to
            `True` and force the execution on the CPU.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments to pass to the relevent `init_process_group` function. Valid `kwargs` can be
            found in [`utils.InitProcessGroupKwargs`]. See the example section for detailed usage.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed. (Choose from 'no','fp16','bf16 or 'fp8').
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
        - **debug** (`bool`) -- Whether or not the current script is being run in debug mode.

    Example:
    ```python
    from accelerate.utils import InitProcessGroupKwargs

    # To include `InitProcessGroupKwargs`, init then call `.to_kwargs()`
    kwargs = InitProcessGroupKwargs(...).to_kwargs()
    state = PartialState(**kwargs)
    ```
    """

    _shared_state = SharedDict()
    _known_attrs = [
        "_cpu",
        "_mixed_precision",
        "_shared_state",
        "backend",
        "debug",
        "device",
        "distributed_type",
        "fork_launched",
        "local_process_index",
        "num_processes",
        "process_index",
    ]

    def __init__(self, cpu: bool = False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None
            self.debug = parse_flag_from_env("ACCELERATE_DEBUG_MODE")
            use_sagemaker_dp = kwargs.pop("_use_sagemaker_dp", None)
            dist_information = None
            if use_sagemaker_dp is None:
                use_sagemaker_dp = (
                    os.environ.get("ACCELERATE_USE_SAGEMAKER", "false") == "true"
                    and os.environ.get("ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE") != SageMakerDistributedType.NO
                )

            # Sets up self.backend + imports
            original_backend = kwargs.pop("backend", None)
            backend, distributed_type = self._prepare_backend(cpu, use_sagemaker_dp, original_backend)
            if original_backend is not None and backend != original_backend:
                raise ValueError(f"Your assigned backend {original_backend} is not avaliable, please use {backend}")
            self.backend = backend
            self.distributed_type = distributed_type
            use_deepspeed = False
            if not cpu and self.backend != "xla":
                if int(os.environ.get("LOCAL_RANK", -1)) != -1:
                    # Deal with spawning deepspeed
                    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
                        if not is_deepspeed_available():
                            raise ImportError(
                                "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                            )
                        from deepspeed import comm as dist

                        if is_xpu_available() and is_ccl_available():
                            os.environ["CCL_PROCESS_LAUNCHER"] = "none"
                            os.environ["CCL_LOCAL_SIZE"] = os.environ.get("LOCAL_WORLD_SIZE", "1")
                            os.environ["CCL_LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")

                        if not dist.is_initialized():
                            dist.init_distributed(dist_backend=self.backend, auto_mpi_discovery=False, **kwargs)
                        # We need to flag to `use_deepspeed` to be True to override `distributed_type` later
                        use_deepspeed = True
                    # Deal with all other backends but XPU and CPU, that gets handled special later
                    elif (
                        self.distributed_type not in (DistributedType.MULTI_XPU, DistributedType.MULTI_CPU)
                        and not torch.distributed.is_initialized()
                    ):
                        torch.distributed.init_process_group(backend=self.backend, **kwargs)
            # XPU and CPU require special env configs to be set
            if self.distributed_type in (DistributedType.MULTI_XPU, DistributedType.MULTI_CPU):
                dist_information = get_cpu_distributed_information()
                os.environ["RANK"] = str(dist_information.rank)
                os.environ["WORLD_SIZE"] = str(dist_information.world_size)
                os.environ["LOCAL_RANK"] = str(dist_information.local_rank)
                os.environ["LOCAL_WORLD_SIZE"] = str(dist_information.local_world_size)
                if self.backend == "ccl" and self.distributed_type == DistributedType.MULTI_XPU:
                    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
                    os.environ["CCL_LOCAL_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]
                    os.environ["CCL_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if (
                    not os.environ.get("MASTER_ADDR", None)
                    and dist_information.local_world_size != dist_information.world_size
                    and self.backend != "mpi"
                ):
                    raise ValueError(
                        "Tried to launch on distributed with multinode, but `MASTER_ADDR` env was not set, "
                        "please try exporting rank 0's hostname as `MASTER_ADDR`"
                    )
                kwargs["rank"] = dist_information.rank
                kwargs["world_size"] = dist_information.world_size

                if (
                    self.distributed_type == DistributedType.MULTI_CPU
                    and get_int_from_env(["OMP_NUM_THREADS"], 0) == 0
                ):
                    import psutil

                    num_cpu_threads_per_process = int(
                        psutil.cpu_count(logical=False) / dist_information.local_world_size
                    )
                    if num_cpu_threads_per_process == 0:
                        num_cpu_threads_per_process = 1
                    torch.set_num_threads(num_cpu_threads_per_process)
                    warnings.warn(
                        f"OMP_NUM_THREADS/MKL_NUM_THREADS unset, we set it at {num_cpu_threads_per_process} to improve oob"
                        " performance."
                    )

                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend=self.backend, **kwargs)

            # No backend == no distributed training
            if self.backend is None:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = 0
                self.local_process_index = 0
            elif self.backend == "xla":
                # XLA needs device setting first for `set_replication`
                self.set_device()
                xm.set_replication(self.device, xm.get_xla_supported_devices())
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                if is_torch_xla_available(check_is_tpu=True):
                    self.local_process_index = xm.get_local_ordinal()
                else:
                    self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
            else:
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = (
                    int(os.environ.get("LOCAL_RANK", -1)) if dist_information is None else dist_information.local_rank
                )
            self.set_device()
            # Now we can change to deepseed
            if use_deepspeed:
                self.distributed_type = DistributedType.DEEPSPEED

            # Set CPU affinity if enabled
            if parse_flag_from_env("ACCELERATE_CPU_AFFINITY", False):
                set_numa_affinity(self.local_process_index)

            # Check for old RTX 4000's that can't use P2P or IB and are on old drivers
            if self.device.type == "cuda" and not check_cuda_p2p_ib_support():
                if "NCCL_P2P_DISABLE" not in os.environ or "NCCL_IB_DISABLE" not in os.environ:
                    raise NotImplementedError(
                        "Using RTX 4000 series doesn't support faster communication broadband via P2P or IB. "
                        'Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which '
                        "will do this automatically."
                    )
        # Important: This should be the *only* code outside of `self.initialized!`
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
        PartialState._shared_state.clear()

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
            DistributedType.MULTI_MLU,
            DistributedType.MULTI_NPU,
            DistributedType.MULTI_XPU,
            DistributedType.MULTI_CPU,
            DistributedType.DEEPSPEED,
            DistributedType.FSDP,
        ):
            torch.distributed.barrier()
        elif self.distributed_type == DistributedType.XLA:
            xm.rendezvous("accelerate.utils.wait_for_everyone")

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool = False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, `dict` of `list`/`tuple`/`torch.Tensor`, or `datasets.Dataset`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
                in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.


        Example:

        ```python
        # Assume there are two processes
        from accelerate import PartialState

        state = PartialState()
        with state.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        if self.num_processes == 1:
            yield inputs
            return
        length = len(inputs)
        # Nested dictionary of any types
        if isinstance(inputs, dict):
            length = len(inputs[list(inputs.keys())[0]])
            if not all(len(v) == length for v in inputs.values()):
                raise ValueError("All values in the dictionary must have the same length")
        num_samples_per_process, num_extras = divmod(length, self.num_processes)
        start_index = self.process_index * num_samples_per_process + min(self.process_index, num_extras)
        end_index = start_index + num_samples_per_process + (1 if self.process_index < num_extras else 0)

        def _split_values(inputs, start_index, end_index):
            if isinstance(inputs, (list, tuple, torch.Tensor)):
                if start_index >= len(inputs):
                    result = inputs[-1:]
                else:
                    result = inputs[start_index:end_index]
                if apply_padding:
                    if isinstance(result, torch.Tensor):
                        from accelerate.utils import pad_across_processes, send_to_device

                        # The tensor needs to be on the device before we can pad it
                        tensorized_result = send_to_device(result, self.device)
                        result = pad_across_processes(tensorized_result, pad_index=inputs[-1])
                    else:
                        result += [result[-1]] * (num_samples_per_process + 1 - len(result))
                return result
            elif isinstance(inputs, dict):
                for key in inputs.keys():
                    inputs[key] = _split_values(inputs[key], start_index, end_index)
                return inputs
            else:
                if is_datasets_available():
                    from datasets import Dataset

                    if isinstance(inputs, Dataset):
                        if start_index >= len(inputs):
                            start_index = len(inputs) - 1
                        if end_index > len(inputs):
                            end_index = len(inputs)
                        result_idcs = list(range(start_index, end_index))
                        if apply_padding:
                            result_idcs += [end_index - 1] * (num_samples_per_process + 1 - len(result_idcs))
                        return inputs.select(result_idcs)
                return inputs

        yield _split_values(inputs, start_index, end_index)

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

    @property
    def default_device(self) -> torch.device:
        """
        Returns the default device which is:
        - MPS if `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()` both return True.
        - CUDA if `torch.cuda.is_available()`
        - MLU if `is_mlu_available()`
        - NPU if `is_npu_available()`
        - CPU otherwise
        """
        if is_mps_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return torch.device("mps")
        elif is_mlu_available():
            return torch.device("mlu")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif is_xpu_available():
            return torch.device("xpu:0")
        elif is_npu_available():
            return torch.device("npu")
        else:
            return torch.device("cpu")

    def _prepare_backend(
        self, cpu: bool = False, sagemaker_dp=False, backend: str = None
    ) -> tuple[str, DistributedType]:
        "Prepares any imports needed before initializing the distributed backend and sets `self.backend` properly"
        distributed_type = None
        if sagemaker_dp:
            import smdistributed.dataparallel.torch.torch_smddp  # noqa

            backend = "smddp"
            distributed_type = DistributedType.MULTI_GPU
        elif is_torch_xla_available():
            backend = "xla"
            distributed_type = DistributedType.XLA
        elif int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
            if is_mlu_available():
                backend = "cncl"
                distributed_type = DistributedType.MULTI_MLU
            elif torch.cuda.is_available():
                if backend is None:
                    backend = "nccl"
                distributed_type = DistributedType.MULTI_GPU
            elif is_npu_available():
                backend = "hccl"
                distributed_type = DistributedType.MULTI_NPU

        if distributed_type is None and (
            int(os.environ.get("LOCAL_RANK", -1)) != -1
            or get_int_from_env(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"], 1) > 1
        ):
            if not cpu and is_xpu_available():
                distributed_type = DistributedType.MULTI_XPU
            else:
                distributed_type = DistributedType.MULTI_CPU

            if (
                backend in (None, "ccl")
                and is_ccl_available()
                and (get_int_from_env(["CCL_WORKER_COUNT"], 0) > 0 or distributed_type == DistributedType.MULTI_XPU)
            ):
                if get_ccl_version() >= "1.12":
                    import oneccl_bindings_for_pytorch  # noqa: F401
                else:
                    import torch_ccl  # noqa: F401

                backend = "ccl"
            elif backend in (None, "mpi") and torch.distributed.is_mpi_available():
                backend = "mpi"
            else:
                backend = "gloo"
        if distributed_type is None:
            distributed_type = DistributedType.NO

        return backend, distributed_type

    def set_device(self):
        """
        Sets the device in `self.device` to the current distributed environment.
        """
        if self.device is not None:
            return
        if self.distributed_type == DistributedType.NO:
            self.device = torch.device("cpu") if self._cpu else self.default_device
            return
        device = str(self.distributed_type).split(".")[-1].replace("MULTI_", "").lower()
        if device not in ("cpu", "gpu", "mlu", "npu", "xpu", "xla"):
            raise ValueError(
                f"Can't set device for {self.distributed_type} ({device}), verify we should be calling `_set_device()` for it!"
            )
        if device == "xla":
            self.device = xm.xla_device()
        else:
            if device == "gpu":
                device = "cuda"
            self.device = torch.device(device, self.local_process_index)
        if self.device is not None:
            if device == "xpu":
                torch.xpu.set_device(self.device)
            elif device == "mlu":
                torch.mlu.set_device(self.device)
            elif device == "npu":
                torch.npu.set_device(self.device)
            elif device == "cuda":
                torch.cuda.set_device(self.device)

    def __getattr__(self, name: str):
        # By this point we know that no attributes of `self` contain `name`,
        # so we just modify the error message
        if name in self._known_attrs:
            raise AttributeError(
                f"`PartialState` object has no attribute `{name}`. "
                "This happens if `PartialState._reset_state()` was called and "
                "an `Accelerator` or `PartialState` was not reinitialized."
            )
        # Raise a typical AttributeError
        raise AttributeError(f"'PartialState' object has no attribute '{name}'")


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
          of mixed precision being performed. (Choose from 'no','fp16','bf16 or 'fp8').
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
        - **debug** (`bool`) -- Whether or not the current script is being run in debug mode.
    """

    _shared_state = SharedDict()
    _known_attrs = PartialState._known_attrs + [
        "deepspeed_plugin",
        "use_ipex",
        "fsdp_plugin",
        "megatron_lm_plugin",
        "dynamo_plugin",
    ]

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
            self.use_ipex = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            if mixed_precision == "fp8":
                if not is_fp8_available():
                    raise ValueError(
                        "Using `fp8` precision requires `transformer_engine` or `MS-AMP` to be installed."
                    )
                elif not check_fp8_capability():
                    logger.warning(
                        f"The current device has compute capability of {torch.cuda.get_device_capability()} which is "
                        "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                        "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                    )
                    mixed_precision = "fp16"

            self.dynamo_plugin = dynamo_plugin
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            # deepspeed handles mixed_precision using deepspeed_config
            self._mixed_precision = "no" if self.distributed_type == DistributedType.DEEPSPEED else mixed_precision
            if self.distributed_type == DistributedType.XLA and is_torch_xla_available(check_is_tpu=True):
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
            elif self.distributed_type in [
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_MLU,
                DistributedType.MULTI_NPU,
                DistributedType.MULTI_XPU,
            ]:
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != "no":
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
                if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true" and self.distributed_type not in [
                    DistributedType.MULTI_XPU,
                ]:
                    self.distributed_type = DistributedType.MEGATRON_LM
                    megatron_lm_plugin.set_mixed_precision(self._mixed_precision)
                    self.megatron_lm_plugin = megatron_lm_plugin
            elif self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
                if is_ipex_available():
                    # check if user disables it explicitly
                    self.use_ipex = parse_flag_from_env("ACCELERATE_USE_IPEX", default=True)
                else:
                    self.use_ipex = False
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
        AcceleratorState._shared_state.clear()
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
    def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool = False):
        """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
                in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.


        Example:

        ```python
        # Assume there are two processes
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        with state.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
        with PartialState().split_between_processes(inputs, apply_padding=apply_padding) as inputs:
            yield inputs

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().main_process_first():
            yield

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        with PartialState().local_main_process_first():
            yield

    def print(self, *args, **kwargs):
        PartialState().print(*args, **kwargs)

    def __getattr__(self, name: str):
        # By this point we know that no attributes of `self` contain `name`,
        # so we just modify the error message
        if name in self._known_attrs:
            raise AttributeError(
                f"`AcceleratorState` object has no attribute `{name}`. "
                "This happens if `AcceleratorState._reset_state()` was called and "
                "an `Accelerator` or `PartialState` was not reinitialized."
            )
        # Raise a typical AttributeError
        raise AttributeError(f"'AcceleratorState' object has no attribute '{name}'")


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
        - **sync_with_dataloader** (`bool`) -- Whether the gradients should be synced at the end of the dataloader
            iteration and the number of total steps reset
        - **is_xla_gradients_synced** (`bool`) -- Whether the XLA gradients have been synchronized. It is initialized
          as false. Once gradients have been reduced before the optimizer step, this flag is set to true. Subsequently,
            after each step, the flag is reset to false. FSDP will always synchronize the gradients, hence
            is_xla_gradients_synced is always true.
    """

    _shared_state = SharedDict()

    def __init__(self, gradient_accumulation_plugin: Optional[GradientAccumulationPlugin] = None):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self.active_dataloader = None
            self.dataloader_references = [None]
            self.plugin_kwargs = (
                gradient_accumulation_plugin.to_kwargs() if gradient_accumulation_plugin is not None else {}
            )
            self._is_xla_gradients_synced = False

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
    def sync_with_dataloader(self) -> bool:
        "Returns whether the gradients should be synced at the end of the dataloader iteration and the number of total steps reset"
        return self.plugin_kwargs.get("sync_with_dataloader", True)

    @property
    def initialized(self) -> bool:
        "Returns whether the `GradientState` has been initialized"
        return GradientState._shared_state != {}

    @property
    def end_of_dataloader(self) -> bool:
        "Returns whether we have reached the end of the current dataloader"
        if not self.in_dataloader:
            return False
        return self.active_dataloader.end_of_dataloader

    @property
    def remainder(self) -> int:
        "Returns the number of extra samples that were added from padding the dataloader"
        if not self.in_dataloader:
            return -1
        return self.active_dataloader.remainder

    def __repr__(self):
        return (
            f"Sync Gradients: {self.sync_gradients}\n"
            f"At end of current dataloader: {self.end_of_dataloader}\n"
            f"Extra samples added: {self.remainder}\n"
            f"Gradient accumulation plugin: {self.plugin_kwargs}\n"
        )

    @property
    def is_xla_gradients_synced(self):
        "Returns the value of is_xla_gradients_synced. FSDP will always synchronize the gradients, hence is_xla_gradients_synced is always true."
        if parse_flag_from_env("ACCELERATE_USE_FSDP", default=False):
            return True
        return self._is_xla_gradients_synced

    @is_xla_gradients_synced.setter
    def is_xla_gradients_synced(self, is_synced):
        "Set the _is_xla_gradients_synced attribute."
        self._is_xla_gradients_synced = is_synced

    def _set_sync_gradients(self, sync_gradients):
        "Private function that sets whether gradients should be synchronized. Users should not have to call this."
        self.sync_gradients = sync_gradients
        # Allow grad-sync to automatically work on TPUs
        if (
            self.sync_gradients
            and is_torch_xla_available(check_is_tpu=True)
            and PartialState().distributed_type == DistributedType.XLA
        ):
            xm.mark_step()

    def _add_dataloader(self, dataloader):
        "Private function that adds a dataloader to `self.dataloader_references` and sets `in_dataloader` to `True`. Users should not have to call this."
        self.active_dataloader = dataloader
        self.dataloader_references.append(self.active_dataloader)

    def _remove_dataloader(self, dataloader):
        "Private function that removes a dataloader from `self.dataloader_references` and sets `in_dataloader` to `False` if there are no more dataloaders. Users should not have to call this."
        self.dataloader_references.remove(dataloader)
        self.active_dataloader = self.dataloader_references[-1]

    @property
    def in_dataloader(self) -> bool:
        "Returns whether the current process is in a dataloader"
        return self.active_dataloader is not None

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        GradientState._shared_state.clear()
