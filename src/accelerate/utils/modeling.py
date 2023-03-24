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

import gc
import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .imports import is_safetensors_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file

WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


logger = logging.getLogger(__name__)


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def dtype_byte_size(dtype: torch.dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        param_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    if value is not None:
        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ == "Int8Params":
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value


def named_module_tensors(module: nn.Module, include_buffers: bool = True, recurse: bool = False):
    """
    A helper function that gathers all the tensors (parameters + buffers) of a given module. If `include_buffers=True`
    it's the same as doing `module.named_parameters(recurse=recurse) + module.named_buffers(recurse=recurse)`.

    Args:
        module (`torch.nn.Module`):
            The module we want the tensors on.
        include_buffer (`bool`, *optional*, defaults to `True`):
            Whether or not to include the buffers in the result.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
    """
    for named_parameter in module.named_parameters(recurse=recurse):
        yield named_parameter

    if include_buffers:
        for named_buffer in module.named_buffers(recurse=recurse):
            yield named_buffer


class FindTiedParametersResult(list):
    """
    This is a subclass of a list to handle backward compatibility for Transformers. Do not rely on the fact this is not
    a list or on the `values` method as in the future this will be removed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def values(self):
        # TODO: at the next Transformers release (4.28.0) issue a deprecation warning here.
        return sum([x[1:] for x in self], [])


def find_tied_parameters(model: nn.Module, **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return FindTiedParametersResult([sorted([weight] + list(set(tied))) for weight, tied in result.items()])


def retie_parameters(model, tied_params):
    """
    Reties tied parameters in a given model if the link was broken (for instance when adding hooks).

    Args:
        model (`torch.nn.Module`):
            The model in which to retie parameters.
        tied_params (`List[List[str]]`):
            A mapping parameter name to tied parameter name as obtained by `find_tied_parameters`.
    """
    for tied_group in tied_params:
        param_to_tie = None
        # First iteration of the loop will set param_to_tie, next ones will tie it to the others
        for param_name in tied_group:
            module = model
            splits = param_name.split(".")
            for split in splits[:-1]:
                module = getattr(module, split)
            if param_to_tie is None:
                param_to_tie = getattr(module, splits[-1])
            else:
                setattr(module, splits[-1], param_to_tie)


def _get_proper_dtype(dtype: Union[str, torch.device]) -> torch.dtype:
    """
    Just does torch.dtype(dtype) if necessary.
    """
    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)
    return dtype


def compute_module_sizes(
    model: nn.Module,
    dtype: Optional[Union[str, torch.device]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.device]]] = None,
):
    """
    Compute the size of each submodule of a given model.
    """
    if dtype is not None:
        dtype = _get_proper_dtype(dtype)
        dtype_size = dtype_byte_size(dtype)
    if special_dtypes is not None:
        special_dtypes = {key: _get_proper_dtype(dtyp) for key, dtyp in special_dtypes.items()}
        special_dtypes_size = {key: dtype_byte_size(dtyp) for key, dtyp in special_dtypes.items()}
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes_size[name]
        elif dtype is None:
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        else:
            size = tensor.numel() * min(dtype_size, dtype_byte_size(tensor.dtype))
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


def get_max_layer_size(
    modules: List[Tuple[str, torch.nn.Module]], module_sizes: Dict[str, int], no_split_module_classes: List[str]
):
    """
    Utility function that will scan a list of named modules and return the maximum size used by one full layer. The
    definition of a layer being:
    - a module with no direct children (just parameters and buffers)
    - a module whose class name is in the list `no_split_module_classes`

    Args:
        modules (`List[Tuple[str, torch.nn.Module]]`):
            The list of named modules where we want to determine the maximum layer size.
        module_sizes (`Dict[str, int]`):
            A dictionary mapping each layer name to its size (as generated by `compute_module_sizes`).
        no_split_module_classes (`List[str]`):
            A list of class names for layers we don't want to be split.

    Returns:
        `Tuple[int, List[str]]`: The maximum size of a layer with the list of layer names realizing that maximum size.
    """
    max_size = 0
    layer_names = []
    modules_to_treat = modules.copy()
    while len(modules_to_treat) > 0:
        module_name, module = modules_to_treat.pop(0)
        modules_children = list(module.named_children()) if isinstance(module, torch.nn.Module) else []
        if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
            # No splitting this one so we compare to the max_size
            size = module_sizes[module_name]
            if size > max_size:
                max_size = size
                layer_names = [module_name]
            elif size == max_size:
                layer_names.append(module_name)
        else:
            modules_to_treat = [(f"{module_name}.{n}", v) for n, v in modules_children] + modules_to_treat
    return max_size, layer_names


def get_max_memory(max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    """
    import psutil

    if max_memory is None:
        if not torch.cuda.is_available():
            max_memory = {}
        else:
            # Make sure CUDA is initialized on each GPU to have the right memory info.
            for i in range(torch.cuda.device_count()):
                _ = torch.tensor([0], device=i)
            max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = psutil.virtual_memory().available
        return max_memory

    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])
    return max_memory


def clean_device_map(device_map: Dict[str, Union[int, str, torch.device]], module_name: str = ""):
    """
    Cleans a device_map by grouping all submodules that go on the same device together.
    """
    # Get the value of the current module and if there is only one split across several keys, regroup it.
    prefix = "" if module_name == "" else f"{module_name}."
    values = [v for k, v in device_map.items() if k.startswith(prefix)]
    if len(set(values)) == 1 and len(values) > 1:
        for k in [k for k in device_map if k.startswith(prefix)]:
            del device_map[k]
        device_map[module_name] = values[0]

    # Recurse over the children
    children_modules = [k for k in device_map.keys() if k.startswith(module_name) and len(k) > len(module_name)]
    idx = len(module_name.split(".")) + 1 if len(module_name) > 0 else 1
    children_modules = set(".".join(k.split(".")[:idx]) for k in children_modules)
    for child in children_modules:
        clean_device_map(device_map, module_name=child)

    return device_map


def load_offloaded_weights(model, index, offload_folder):
    """
    Loads the weights from the offload folder into the model.

    Args:
        model (`torch.nn.Module`):
            The model to load the weights into.
        index (`dict`):
            A dictionary containing the parameter name and its metadata for each parameter that was offloaded from the
            model.
        offload_folder (`str`):
            The folder where the offloaded weights are stored.
    """
    if index is None or len(index) == 0:
        # Nothing to do
        return

    for param_name, metadata in index.items():
        tensor_file = os.path.join(offload_folder, f"{param_name}.dat")
        weight = load_offloaded_weight(tensor_file, metadata)
        set_module_tensor_to_device(model, param_name, "cpu", value=weight)


def get_balanced_memory(
    model: nn.Module,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.device]]] = None,
    low_zero: bool = False,
):
    """
    Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        low_zero (`bool`, *optional*):
            Minimizes the number of weights on GPU 0, which is convenient when it's used for other operations (like the
            Transformers generate function).
    """
    # Get default / clean up max_memory
    max_memory = get_max_memory(max_memory)

    if not torch.cuda.is_available():
        return max_memory

    num_devices = len([d for d in max_memory if torch.device(d).type == "cuda" and max_memory[d] > 0])
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    per_gpu = module_sizes[""] // (num_devices - 1 if low_zero else num_devices)

    # We can't just set the memory to model_size // num_devices as it will end being too small: each GPU will get
    # slightly less layers and some layers will end up offload at the end. So this function computes a buffer size to
    # add which is the biggest of:
    # - the size of no split block (if applicable)
    # - the mean of the layer sizes
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    # Identify the size of the no_split_block modules
    if len(no_split_module_classes) > 0:
        no_split_children = {}
        for name, size in module_sizes.items():
            if name == "":
                continue
            submodule = model
            for submodule_name in name.split("."):
                submodule = getattr(submodule, submodule_name)
            class_name = submodule.__class__.__name__
            if class_name in no_split_module_classes and class_name not in no_split_children:
                no_split_children[class_name] = size

            if set(no_split_children.keys()) == set(no_split_module_classes):
                break
        buffer = max(no_split_children.values()) if len(no_split_children) > 0 else 0
    else:
        buffer = 0

    # Compute mean of final modules. In the first dict of module sizes, leaves are the parameters
    leaves = [n for n in module_sizes if len([p for p in module_sizes if p.startswith(n) and len(p) > len(n)]) == 0]
    module_sizes = {n: v for n, v in module_sizes.items() if n not in leaves}
    # Once removed, leaves are the final modules.
    leaves = [n for n in module_sizes if len([p for p in module_sizes if p.startswith(n) and len(p) > len(n)]) == 0]
    mean_leaves = int(sum([module_sizes[n] for n in leaves]) / len(leaves))
    buffer = int(1.25 * max(buffer, mean_leaves))
    per_gpu += buffer

    max_memory = get_max_memory(max_memory)
    last_gpu = max(i for i in max_memory if isinstance(i, int) and max_memory[i] > 0)
    # The last device is left with max_memory just in case the buffer is not enough.
    for i in range(last_gpu):
        max_memory[i] = min(0 if low_zero and i == 0 else per_gpu, max_memory[i])

    if low_zero:
        min_zero = max(0, module_sizes[""] - sum([max_memory[i] for i in range(1, num_devices)]))
        max_memory[0] = min(min_zero, max_memory[0])

    return max_memory


def infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.dtype]]] = None,
    verbose: bool = False,
):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
    """
    # Get default / clean up max_memory
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    devices = list(max_memory.keys())
    gpus = [device for device in devices if device != "cpu"]
    if "disk" not in devices:
        devices.append("disk")

    # Devices that need to keep space for a potential offloaded layer.
    main_devices = [gpus[0], "cpu"] if len(gpus) > 0 else ["cpu"]

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)

    device_map = {}
    current_device = 0
    current_memory_used = 0

    # Direct submodules and parameters
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )
    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if not n.startswith(name)]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                module_sizes,
                no_split_module_classes,
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        tied_param_goups = [
            tied_group
            for tied_group in tied_parameters
            if any(name in k for k in tied_group) and not all(name in k for k in tied_group)
        ]
        if verbose and len(tied_param_goups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_goups}")
        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum([[p for p in tied_group if name not in p] for tied_group in tied_param_goups], [])
        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
        # Case 1 -> We're too big!
        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            # Split or not split?
            modules_children = list(module.named_children())
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size-current_memory_used}, module size {module_size})."
                )
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                # -> no split, we go to the next device
                if verbose:
                    print("This module cannot be split, going to the next device.")
                current_device += 1
                modules_to_treat = [(name, module)] + modules_to_treat
                current_memory_used = 0
            else:
                # -> split, we replace the module studied by its children + parameters
                if verbose:
                    print(f"Splitting {name}.")
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_treat
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )

        # Case 2, it fits! We're not entirely out of the wood though, because we may have some tied parameters.
        elif len(tied_params) > 0:
            # First locate all tied modules
            tied_module_names = []
            tied_modules = []
            for tied_param in tied_params:
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n in tied_param][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])
            if verbose:
                print(
                    f"  It looks like {name} is going to fit on {devices[current_device]} but we have tied "
                    f"parameters to account for.\n  - Names {tied_params}\n  - Module names {tied_module_names}"
                )

            # Let's see if it all fits first
            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

            if current_max_size is None or current_memory_used + module_size_with_ties <= current_max_size:
                # We really really fit!
                if verbose:
                    print(f"Putting {name} and {tied_module_names} on {devices[current_device]}.")
                current_memory_used += module_size_with_ties
                device_map[name] = devices[current_device]
                for tied_module_name in tied_module_names:
                    tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]
                    modules_to_treat.pop(tied_module_index)
                    device_map[tied_module_name] = devices[current_device]

            else:
                # We don't fit with the tied modules. Next question is: can we split one of the tied modules to make it
                # smaller or do we need to go on the next device?
                if verbose:
                    print(
                        f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                        f"available {current_max_size-current_memory_used}, needed size {module_size_with_ties})."
                    )
                split_happened = False
                for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                    tied_module_children = list(tied_module.named_children())
                    if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                        # can't break this one.
                        continue

                    if verbose:
                        print(f"Splitting {tied_module_name}.")
                    tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                    tied_module_children = [(f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                    tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                    modules_to_treat = (
                        [(name, module)]
                        + modules_to_treat[:tied_module_index]
                        + tied_module_children
                        + modules_to_treat[tied_module_index + 1 :]
                    )
                    # Update the max layer size.
                    max_layer_size, max_layer_names = get_max_layer_size(
                        [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                        module_sizes,
                        no_split_module_classes,
                    )
                    split_happened = True
                    break

                if not split_happened:
                    # If the tied module is not split, we go to the next device
                    if verbose:
                        print("None of the tied module can be split, going to the next device.")
                    current_device += 1
                    modules_to_treat = [(name, module)] + modules_to_treat
                    current_memory_used = 0

        else:
            if verbose:
                print(f"Putting {name} on {devices[current_device]}.")
            current_memory_used += module_size
            device_map[name] = devices[current_device]

    return clean_device_map(device_map)


def check_device_map(model: nn.Module, device_map: Dict[str, Union[int, str, torch.device]]):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        all_model_tensors = [name for name in all_model_tensors if not name.startswith(module_name)]
    if len(all_model_tensors) > 0:
        non_covered_params = ", ".join(all_model_tensors)
        raise ValueError(
            f"The device_map provided does not give any device for the following parameters: {non_covered_params}"
        )


def load_state_dict(checkpoint_file, device_map=None):
    """
    Load a checkpoint from a given file. If the checkpoint is in the safetensors format and a device map is passed, the
    weights can be fast-loaded directly on the GPU.

    Args:
        checkpoint_file (`str`): The path to the checkpoint to load.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
    """
    if checkpoint_file.endswith(".safetensors"):
        if not is_safetensors_available():
            raise ImportError(
                f"To load {checkpoint_file}, the `safetensors` library is necessary `pip install safetensors`."
            )
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
            weight_names = f.keys()

        if metadata is None:
            logger.warn(
                f"The safetensors archive passed at {checkpoint_file} does not contain metadata. "
                "Make sure to save your model with the `save_pretrained` method. Defaulting to 'pt' metadata."
            )
            metadata = {"format": "pt"}

        if metadata.get("format") not in ["pt", "tf", "flax"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        elif metadata["format"] != "pt":
            raise ValueError(f"The checkpoint passed was saved with {metadata['format']}, we need a the pt format.")
        if device_map is None:
            return safe_load_file(checkpoint_file)
        else:
            devices = [device for device in device_map.values() if device not in ["disk"]]

            # if we only have one device we can load everything directly
            if len(devices) == 1:
                return safe_load_file(checkpoint_file, device=devices[0])

            # cpu device should always exist as fallback option
            if "cpu" not in devices:
                devices.append("cpu")

            # For each device, get the weights that go there
            device_weights = {device: [] for device in devices}
            for module_name, device in device_map.items():
                if device in devices:
                    device_weights[device].extend([k for k in weight_names if k.startswith(module_name)])

            # all weights that haven't defined a device should be loaded on CPU
            device_weights["cpu"].extend([k for k in weight_names if k not in sum(device_weights.values(), [])])
            tensors = {}
            for device in devices:
                with safe_open(checkpoint_file, framework="pt", device=device) as f:
                    for key in device_weights[device]:
                        tensors[key] = f.get_tensor(key)

            return tensors
    else:
        return torch.load(checkpoint_file)


def load_checkpoint_in_model(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: bool = False,
    offload_buffers: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded.

    <Tip warning={true}>

    Once loaded across devices, you still need to call [`dispatch_model`] on your model to make it able to run. To
    group the checkpoint loading and dispatch in one single call, use [`load_checkpoint_and_dispatch`].

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.
        offload_buffers (`bool`, *optional*, defaults to `False):
            Whether or not to include the buffers in the weights offloaded to disk.
    """
    tied_params = find_tied_parameters(model)
    if offload_folder is None and device_map is not None and "disk" in device_map.values():
        raise ValueError(
            "At least one of the model submodule will be offloaded to disk, please pass along an `offload_folder`."
        )
    elif offload_folder is not None and device_map is not None and "disk" in device_map.values():
        os.makedirs(offload_folder, exist_ok=True)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
        if len(potential_index) == 0:
            raise ValueError(f"{checkpoint} is not a folder containing a `.index.json` file.")
        elif len(potential_index) == 1:
            index_filename = os.path.join(checkpoint, potential_index[0])
        else:
            raise ValueError(f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones.")
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.

    offload_index = {}
    if offload_state_dict:
        state_dict_folder = tempfile.mkdtemp()
        state_dict_index = {}

    buffer_names = [name for name, _ in model.named_buffers()]

    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=device_map)
        if device_map is None:
            model.load_state_dict(checkpoint, strict=False)
        else:
            for param_name, param in checkpoint.items():
                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=dtype)

        # Force Python to clean up.
        del checkpoint
        gc.collect()

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)
