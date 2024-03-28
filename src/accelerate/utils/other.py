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

import collections
import os
import platform
import re
import socket
from contextlib import contextmanager
from functools import partial, reduce
from types import MethodType
from typing import OrderedDict

import torch
from packaging.version import Version
from safetensors.torch import save_file as safe_save_file

from ..commands.config.default import write_basic_config  # noqa: F401
from ..logging import get_logger
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_torch_distributed_available, is_torch_xla_available
from .modeling import id_tensor_storage
from .transformer_engine import convert_model
from .versions import is_torch_version


logger = get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


def is_compiled_module(module):
    """
    Check whether the module was compiled with torch.compile()
    """
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True, recursive: bool = False):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.

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

    if is_torch_version(">=", FSDP_PYTORCH_VERSION) and is_torch_distributed_available():
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        options += (FSDP,)

    while isinstance(model, options):
        model = model.module

    if recursive:
        # This is needed in cases such as using FSDPv2 on XLA
        def _recursive_unwrap(module):
            # Wrapped modules are standardly wrapped as `module`, similar to the cases earlier
            # with DDP, DataParallel, DeepSpeed, and FSDP
            if hasattr(module, "module"):
                unwrapped_module = _recursive_unwrap(module.module)
            else:
                unwrapped_module = module
            # Next unwrap child sublayers recursively
            for name, child in unwrapped_module.named_children():
                setattr(unwrapped_module, name, _recursive_unwrap(child))
            return unwrapped_module

        # Start with top-level
        model = _recursive_unwrap(model)

    if not keep_fp32_wrapper:
        forward = model.forward
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


def clean_state_dict_for_safetensors(state_dict: dict):
    """
    Cleans the state dictionary from a model and removes tensor aliasing if present.

    Args:
        state_dict (`dict`):
            The state dictionary from a model
    """
    ptrs = collections.defaultdict(list)
    # When bnb serialization is used, weights in state dict can be strings
    for name, tensor in state_dict.items():
        if not isinstance(tensor, str):
            ptrs[id_tensor_storage(tensor)].append(name)

    # These are all pointers of tensors with shared memory
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        # When not all duplicates have been cleaned, we still remove those keys but put a clear warning.
        # If the link between tensors was done at runtime then `from_pretrained` will not get
        # the key back leading to random tensor. A proper warning will be shown
        # during reload (if applicable), but since the file is not necessarily compatible with
        # the config, better show a proper warning.
        found_names = [name for name in names if name in state_dict]
        warn_names.update(found_names[1:])
        for name in found_names[1:]:
            del state_dict[name]
    if len(warn_names) > 0:
        logger.warning(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    state_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    return state_dict


def save(obj, f, save_on_each_node: bool = False, safe_serialization: bool = False):
    """
    Save the data to disk. Use in place of `torch.save()`.

    Args:
        obj:
            The data to save
        f:
            The file (or file-like object) to use to save the data
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            Whether to only save on the global main process
        safe_serialization (`bool`, *optional*, defaults to `False`):
            Whether to save `obj` using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    # When TorchXLA is enabled, it's necessary to transfer all data to the CPU before saving.
    # Another issue arises with `id_tensor_storage`, which treats all XLA tensors as identical.
    # If tensors remain on XLA, calling `clean_state_dict_for_safetensors` will result in only
    # one XLA tensor remaining.
    if PartialState().distributed_type == DistributedType.XLA:
        obj = xm._maybe_convert_to_cpu(obj)
    # Check if it's a model and remove duplicates
    if safe_serialization:
        save_func = partial(safe_save_file, metadata={"format": "pt"})
        if isinstance(obj, OrderedDict):
            obj = clean_state_dict_for_safetensors(obj)
    else:
        save_func = torch.save

    if PartialState().is_main_process and not save_on_each_node:
        save_func(obj, f)
    elif PartialState().is_local_main_process and save_on_each_node:
        save_func(obj, f)


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


def check_os_kernel():
    """Warns if the kernel version is below the recommended minimum on Linux."""
    # see issue #1929
    info = platform.uname()
    system = info.system
    if system != "Linux":
        return

    _, version, *_ = re.split(r"(\d+\.\d+\.\d+)", info.release)
    min_version = "5.5.0"
    if Version(version) < Version(min_version):
        msg = (
            f"Detected kernel version {version}, which is below the recommended minimum of {min_version}; this can "
            "cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher."
        )
        logger.warning(msg, main_process_only=True)


def recursive_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return reduce(_getattr, [obj] + attr.split("."))
