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
import platform
import re
import socket
from codecs import encode
from collections import OrderedDict
from functools import partial, reduce
from types import MethodType

import numpy as np
import torch
from packaging.version import Version
from safetensors.torch import save_file as safe_save_file

from ..commands.config.default import write_basic_config  # noqa: F401
from ..logging import get_logger
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import (
    is_deepspeed_available,
    is_numpy_available,
    is_torch_distributed_available,
    is_torch_xla_available,
    is_weights_only_available,
)
from .modeling import id_tensor_storage
from .transformer_engine import convert_model
from .versions import is_torch_version


logger = get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


def is_compiled_module(module: torch.nn.Module) -> bool:
    """
    Check whether the module was compiled with torch.compile()
    """
    if not hasattr(torch, "_dynamo"):
        return False

    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def has_compiled_regions(module: torch.nn.Module) -> bool:
    """
    Check whether the module has submodules that were compiled with `torch.compile()`.
    """
    if not hasattr(torch, "_dynamo"):
        return False

    if module._modules:
        for submodule in module.modules():
            if isinstance(submodule, torch._dynamo.eval_frame.OptimizedModule):
                return True

    return False


def is_repeated_blocks(module: torch.nn.Module) -> bool:
    """
    Check whether the module is a repeated block, i.e. `torch.nn.ModuleList` with all children of the same class. This
    is useful to determine whether we should apply regional compilation to the module.
    """

    return isinstance(module, torch.nn.ModuleList) and all(isinstance(m, module[0].__class__) for m in module)


def has_repeated_blocks(module: torch.nn.Module) -> bool:
    """
    Check whether the module has repeated blocks, i.e. `torch.nn.ModuleList` with all children of the same class, at
    any level of the module hierarchy. This is useful to determine whether we should apply regional compilation to the
    module.
    """
    if module._modules:
        for submodule in module.modules():
            if is_repeated_blocks(submodule):
                return True

    return False


def compile_regions(module: torch.nn.Module, **compile_kwargs) -> torch.nn.Module:
    """
    Performs regional compilation where we target repeated blocks of the same class and compile them sequentially to
    hit the compiler's cache. For example, in `GPT2LMHeadModel`, the repeated block/class is `GPT2Block`, and can be
    accessed as `model.transformer.h[0]`. The rest of the model (e.g. model.lm_head) is compiled separately.

    This allows us to speed up the compilation overhead / cold start of models like LLMs and Transformers in general.
    See https://pytorch.org/tutorials/recipes/regional_compilation.html for more details.

    Args:
        module (`torch.nn.Module`):
            The model to compile.
        **compile_kwargs:
            Additional keyword arguments to pass to `torch.compile()`.

    Returns:
        `torch.nn.Module`: A new instance of the model with some compiled regions.

    Example:
    ```python
    >>> from accelerate.utils import compile_regions
    >>> from transformers import AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> compiled_model = compile_regions(model, mode="reduce-overhead")
    >>> compiled_model.transformer.h[0]
    OptimizedModule(
        (_orig_mod): GPT2Block(
                (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (attn): GPT2Attention(
                (c_attn): Conv1D(nf=2304, nx=768)
                (c_proj): Conv1D(nf=768, nx=768)
                (attn_dropout): Dropout(p=0.1, inplace=False)
                (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
                (c_fc): Conv1D(nf=3072, nx=768)
                (c_proj): Conv1D(nf=768, nx=3072)
                (act): NewGELUActivation()
                (dropout): Dropout(p=0.1, inplace=False)
            )
        )
    )
    ```
    """

    def _compile_regions(module: torch.nn.Module, **compile_kwargs) -> torch.nn.Module:
        if is_repeated_blocks(module):
            new_module = torch.nn.ModuleList()
            for submodule in module:
                new_module.append(torch.compile(submodule, **compile_kwargs))
        elif has_repeated_blocks(module):
            new_module = module.__class__.__new__(module.__class__)
            new_module.__dict__.update(module.__dict__)
            new_module._modules = {}
            for name, submodule in module.named_children():
                new_module.add_module(name, _compile_regions(submodule, **compile_kwargs))
        else:
            new_module = torch.compile(module, **compile_kwargs)

        return new_module

    new_module = _compile_regions(module, **compile_kwargs)

    if "_orig_mod" not in new_module.__dict__:
        # Keeps a reference to the original module to decompile/unwrap it later
        new_module.__dict__["_orig_mod"] = module

    return new_module


def compile_regions_deepspeed(module: torch.nn.Module, **compile_kwargs):
    """
    Performs regional compilation the same way as `compile_regions`, but specifically for `DeepSpeedEngine.module`.
    Since the model is wrapped in a `DeepSpeedEngine` and has many added hooks, offloaded parameters, etc that
    `torch.compile(...)` interferes with, version of trgional compilation uses the inplace `module.compile()` method
    instead.

    Args:
        module (`torch.nn.Module`):
            The model to compile.
        **compile_kwargs:
            Additional keyword arguments to pass to `module.compile()`.
    """

    if is_repeated_blocks(module):
        for submodule in module:
            submodule.compile(**compile_kwargs)
    elif has_repeated_blocks(module):
        for child in module.children():
            compile_regions_deepspeed(child, **compile_kwargs)
    else:  # leaf node
        module.compile(**compile_kwargs)


def model_has_dtensor(model: torch.nn.Module) -> bool:
    """
    Check if the model has DTensor parameters.

    Args:
        model (`torch.nn.Module`):
            The model to check.

    Returns:
        `bool`: Whether the model has DTensor parameters.
    """
    if is_torch_version(">=", "2.5.0"):
        from torch.distributed.tensor import DTensor
    else:
        # from torch 2.0.0 (oldest supported accelerate torch version), DTensor is in torch.distributed._tensor
        from torch.distributed._tensor import DTensor

    return any(isinstance(p, DTensor) for p in model.parameters())


def extract_model_from_parallel(
    model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True, recursive: bool = False
):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.
        keep_torch_compile (`bool`, *optional*):
            Whether to unwrap compiled model.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    is_compiled = is_compiled_module(model)
    has_compiled = has_compiled_regions(model)

    if is_compiled:
        compiled_model = model
        model = model._orig_mod
    elif has_compiled:
        compiled_model = model
        model = model.__dict__["_orig_mod"]

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

    if keep_torch_compile:
        if is_compiled:
            compiled_model._orig_mod = model
            model = compiled_model
        elif has_compiled:
            compiled_model.__dict__["_orig_mod"] = model
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


# The following are considered "safe" globals to reconstruct various types of objects when using `weights_only=True`
# These should be added and then removed after loading in the file
np_core = np._core if is_numpy_available("2.0.0") else np.core
TORCH_SAFE_GLOBALS = [
    # numpy arrays are just numbers, not objects, so we can reconstruct them safely
    np_core.multiarray._reconstruct,
    np.ndarray,
    # The following are needed for the RNG states
    encode,
    np.dtype,
]

if is_numpy_available("1.25.0"):
    TORCH_SAFE_GLOBALS.append(np.dtypes.UInt32DType)


def load(f, map_location=None, **kwargs):
    """
    Compatible drop-in replacement of `torch.load()` which allows for `weights_only` to be used if `torch` version is
    2.4.0 or higher. Otherwise will ignore the kwarg.

    Will also add (and then remove) an exception for numpy arrays

    Args:
        f:
            The file (or file-like object) to use to load the data
        map_location:
            a function, `torch.device`, string or a dict specifying how to remap storage locations
        **kwargs:
            Additional keyword arguments to pass to `torch.load()`.
    """
    try:
        if is_weights_only_available():
            old_safe_globals = torch.serialization.get_safe_globals()
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = True
            torch.serialization.add_safe_globals(TORCH_SAFE_GLOBALS)
        else:
            kwargs.pop("weights_only", None)
        loaded_obj = torch.load(f, map_location=map_location, **kwargs)
    finally:
        if is_weights_only_available():
            torch.serialization.clear_safe_globals()
            if old_safe_globals:
                torch.serialization.add_safe_globals(old_safe_globals)
    return loaded_obj


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


def get_free_port() -> int:
    """
    Gets a free port on `localhost`. Useful for automatic port selection when port 0 is specified in distributed
    training scenarios.

    Returns:
        int: An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to port 0 for OS to assign a free port
        return s.getsockname()[1]


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


def get_module_children_bottom_up(model: torch.nn.Module, return_fqns: bool = False) -> list[torch.nn.Module]:
    """Traverse the model in bottom-up order and return the children modules in that order.

    Args:
        model (`torch.nn.Module`): the model to get the children of

    Returns:
        `list[torch.nn.Module]`: a list of children modules of `model` in bottom-up order. The last element is the
        `model` itself.
    """
    top = model if not return_fqns else ("", model)
    stack = [top]
    ordered_modules = []
    while stack:
        current_module = stack.pop()
        if return_fqns:
            current_module_name, current_module = current_module
        for name, attr in current_module.named_children():
            if isinstance(attr, torch.nn.Module):
                if return_fqns:
                    child_name = current_module_name + "." + name if current_module_name else name
                    stack.append((child_name, attr))
                else:
                    stack.append(attr)
        if return_fqns:
            ordered_modules.append((current_module_name, current_module))
        else:
            ordered_modules.append(current_module)
    return ordered_modules[::-1]
