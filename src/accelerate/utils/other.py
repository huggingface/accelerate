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
import socket
from contextlib import contextmanager
from types import MethodType

import torch

from ..commands.config.default import write_basic_config  # noqa: F401
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_safetensors_available, is_tpu_available
from .transformer_engine import convert_model
from .versions import is_torch_version


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_safetensors_available():
    from safetensors.torch import save_file as safe_save_file


def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod

    if is_deepspeed_available():
        from deepspeed import DeepSpeedEngine

        options += (DeepSpeedEngine,)

    if is_torch_version(">=", FSDP_PYTORCH_VERSION):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        options += (FSDP,)

    while isinstance(model, options):
        model = model.module

    if not keep_fp32_wrapper:
        forward = getattr(model, "forward")
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = MethodType(forward, model)
        if getattr(model, "_converted_to_transformer_engine", False):
            convert_model(model, to_transformer_engine=False)

    if is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model

    return model


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    <Tip warning={true}>

    Make sure all processes will reach this instruction otherwise one of your processes will hang forever.

    </Tip>
    """
    PartialState().wait_for_everyone()


def save(obj, f, safe_serialization=False):
    """
    Save the data to disk. Use in place of `torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
        safe_serialization (`bool`, *optional*, defaults to `False`): Whether to save `obj` using `safetensors`
    """
    if PartialState().distributed_type == DistributedType.TPU:
        xm.save(obj, f)
    elif PartialState().local_process_index == 0:
        if safe_serialization:
            safe_save_file(obj, f, metadata={"format": "pt"})
        else:
            torch.save(obj, f)


@contextmanager
def clear_environment():
    """
    A context manager that will cache origin `os.environ` and replace it with a empty dictionary in this context.

    When this context exits, the cached `os.environ` will be back.

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
    _old_os_environ = os.environ
    os.environ = dict()

    yield

    os.environ = _old_os_environ


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

    yield

    for key in kwargs:
        key = key.upper()
        if key in existing_vars:
            # restore previous value
            os.environ[key] = existing_vars[key]
        else:
            os.environ.pop(key, None)


def get_pretty_name(obj):
    """
    Gets a pretty name from `obj`.
    """
    if not hasattr(obj, "__qualname__") and not hasattr(obj, "__name__"):
        obj = getattr(obj, "__class__", obj)
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)


def merge_dicts(source, destination):
    """
    Recursively merges two dictionaries.

    Args:
        source (`dict`): The dictionary to merge into `destination`.
        destination (`dict`): The dictionary to merge `source` into.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dicts(value, node)
        else:
            destination[key] = value

    return destination


def is_port_in_use(port: int = None) -> bool:
    """
    Checks if a port is in use on `localhost`. Useful for checking if multiple `accelerate launch` commands have been
    run and need to see if the port is already in use.
    """
    if port is None:
        port = 29500
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def convert_bytes(size):
    "Converts `size` from bytes to the largest possible unit"
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{round(size, 2)} {x}"
        size /= 1024.0

    return f"{round(size, 2)} PB"
