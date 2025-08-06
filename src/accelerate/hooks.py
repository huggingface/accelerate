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

import functools
from collections.abc import Mapping
from typing import Optional, Union

import torch
import torch.nn as nn

from .state import PartialState
from .utils import (
    PrefixedDataset,
    find_device,
    named_module_tensors,
    send_to_device,
    set_module_tensor_to_device,
)
from .utils.imports import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
)
from .utils.memory import clear_device_cache
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr


_accelerate_added_attributes = ["to", "cuda", "npu", "xpu", "mlu", "sdaa", "musa"]


class ModelHook:
    """
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.

    Class attribute:
    - **no_grad** (`bool`, *optional*, defaults to `False`) -- Whether or not to execute the actual forward pass under
      the `torch.no_grad()` context manager.
    """

    no_grad = False

    def init_hook(self, module):
        """
        To be executed when the hook is attached to the module.

        Args:
            module (`torch.nn.Module`): The module attached to this hook.
        """
        return module

    def pre_forward(self, module, *args, **kwargs):
        """
        To be executed just before the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`): The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`): The keyword arguments passed to the module.

        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`: A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module, output):
        """
        To be executed just after the forward method of the model.

        Args:
            module (`torch.nn.Module`): The module whose forward pass been executed just before this event.
            output (`Any`): The output of the module.

        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module):
        """
        To be executed when the hook is detached from a module.

        Args:
            module (`torch.nn.Module`): The module detached from this hook.
        """
        return module


class SequentialHook(ModelHook):
    """
    A hook that can contain several hooks and iterates through them at each event.
    """

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in self.hooks:
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module


def add_hook_to_module(module: nn.Module, hook: ModelHook, append: bool = False):
    """
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.

    Returns:
        `torch.nn.Module`: The same module, with the hook attached (the module is modified in place, so the result can
        be discarded).
    """
    if append and (getattr(module, "_hf_hook", None) is not None):
        old_hook = module._hf_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_hf_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._hf_hook = hook

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = module._old_forward(*args, **kwargs)
        else:
            output = module._old_forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)

    # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
    # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
    if "GraphModuleImpl" in str(type(module)):
        module.__class__.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)
    else:
        module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module


def remove_hook_from_module(module: nn.Module, recurse=False):
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`torch.nn.Module`): The module to attach a hook to.
        recurse (`bool`, **optional**): Whether to remove the hooks recursively

    Returns:
        `torch.nn.Module`: The same module, with the hook detached (the module is modified in place, so the result can
        be discarded).
    """

    if hasattr(module, "_hf_hook"):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

    if hasattr(module, "_old_forward"):
        # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(module)):
            module.__class__.forward = module._old_forward
        else:
            module.forward = module._old_forward
        delattr(module, "_old_forward")

    # Remove accelerate added warning hooks from dispatch_model
    for attr in _accelerate_added_attributes:
        module.__dict__.pop(attr, None)

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


class AlignDevicesHook(ModelHook):
    """
    A generic `ModelHook` that ensures inputs and model weights are on the same device for the forward pass of the
    associated module, potentially offloading the weights after the forward pass.

    Args:
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        io_same_device (`bool`, *optional*, defaults to `False`):
            Whether or not the output should be placed on the same device as the input was.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        place_submodules (`bool`, *optional*, defaults to `False`):
            Whether to place the submodules on `execution_device` during the `init_hook` event.
    """

    def __init__(
        self,
        execution_device: Optional[Union[int, str, torch.device]] = None,
        offload: bool = False,
        io_same_device: bool = False,
        weights_map: Optional[Mapping] = None,
        offload_buffers: bool = False,
        place_submodules: bool = False,
        skip_keys: Optional[Union[str, list[str]]] = None,
        tied_params_map: Optional[dict[int, dict[torch.device, torch.Tensor]]] = None,
    ):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys

        # Will contain the input device when `io_same_device=True`.
        self.input_device = None
        self.param_original_devices = {}
        self.buffer_original_devices = {}
        self.tied_params_names = set()

        # The hook pre_forward/post_forward need to have knowledge of this dictionary, as with offloading we want to avoid duplicating memory
        # for tied weights already loaded on the target execution device.
        self.tied_params_map = tied_params_map

    def __repr__(self):
        return (
            f"AlignDevicesHook(execution_device={self.execution_device}, offload={self.offload}, "
            f"io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, "
            f"place_submodules={self.place_submodules}, skip_keys={repr(self.skip_keys)})"
        )

    def init_hook(self, module):
        # In case the AlignDevicesHook is on meta device, ignore tied weights as data_ptr() is then always zero.
        if self.execution_device == "meta" or self.execution_device == torch.device("meta"):
            self.tied_params_map = None

        if not self.offload and self.execution_device is not None:
            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
                set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
        elif self.offload:
            self.original_devices = {
                name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)
            }
            if self.weights_map is None:
                self.weights_map = {
                    name: param.to("cpu")
                    for name, param in named_module_tensors(
                        module, include_buffers=self.offload_buffers, recurse=self.place_submodules
                    )
                }
            for name, _ in named_module_tensors(
                module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True
            ):
                # When using disk offloading, we can not rely on `weights_map[name].data_ptr()` as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.
                # As we have no reliable way to track the shared data pointer of tied weights in this case, we use tied_params_names: List[str]
                # to add on the fly pointers to `tied_params_map` in the pre_forward call.
                if (
                    self.tied_params_map is not None
                    and recursive_getattr(module, name).data_ptr() in self.tied_params_map
                ):
                    self.tied_params_names.add(name)

                set_module_tensor_to_device(module, name, "meta")

            if not self.offload_buffers and self.execution_device is not None:
                for name, _ in module.named_buffers(recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )
            elif self.offload_buffers and self.execution_device is not None:
                for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
                    set_module_tensor_to_device(
                        module, name, self.execution_device, tied_params_map=self.tied_params_map
                    )

        return module

    def pre_forward(self, module, *args, **kwargs):
        if self.io_same_device:
            self.input_device = find_device([args, kwargs])
        if self.offload:
            self.tied_pointers_to_remove = set()

            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                fp16_statistics = None
                value = self.weights_map[name]
                if "weight" in name and name.replace("weight", "SCB") in self.weights_map.keys():
                    if value.dtype == torch.int8:
                        fp16_statistics = self.weights_map[name.replace("weight", "SCB")]

                # In case we are using offloading with tied weights, we need to keep track of the offloaded weights
                # that are loaded on device at this point, as we will need to remove them as well from the dictionary
                # self.tied_params_map in order to allow to free memory.
                if name in self.tied_params_names and value.data_ptr() not in self.tied_params_map:
                    self.tied_params_map[value.data_ptr()] = {}

                if (
                    value is not None
                    and self.tied_params_map is not None
                    and value.data_ptr() in self.tied_params_map
                    and self.execution_device not in self.tied_params_map[value.data_ptr()]
                ):
                    self.tied_pointers_to_remove.add((value.data_ptr(), self.execution_device))

                set_module_tensor_to_device(
                    module,
                    name,
                    self.execution_device,
                    value=value,
                    fp16_statistics=fp16_statistics,
                    tied_params_map=self.tied_params_map,
                )

        return send_to_device(args, self.execution_device), send_to_device(
            kwargs, self.execution_device, skip_keys=self.skip_keys
        )

    def post_forward(self, module, output):
        if self.offload:
            for name, _ in named_module_tensors(
                module,
                include_buffers=self.offload_buffers,
                recurse=self.place_submodules,
                remove_non_persistent=True,
            ):
                set_module_tensor_to_device(module, name, "meta")
                if type(module).__name__ == "Linear8bitLt":
                    module.state.SCB = None
                    module.state.CxB = None

            # We may have loaded tied weights into self.tied_params_map (avoiding to load them several times in e.g. submodules): remove them from
            # this dictionary to allow the garbage collector to do its job.
            for value_pointer, device in self.tied_pointers_to_remove:
                if isinstance(device, int):
                    if is_npu_available():
                        device = f"npu:{device}"
                    elif is_mlu_available():
                        device = f"mlu:{device}"
                    elif is_musa_available():
                        device = f"musa:{device}"
                if device in self.tied_params_map[value_pointer]:
                    del self.tied_params_map[value_pointer][device]
            self.tied_pointers_to_remove = set()
        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device, skip_keys=self.skip_keys)

        return output

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device("meta"):
                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))
        return module


def attach_execution_device_hook(
    module: torch.nn.Module,
    execution_device: Union[int, str, torch.device],
    skip_keys: Optional[Union[str, list[str]]] = None,
    preload_module_classes: Optional[list[str]] = None,
    tied_params_map: Optional[dict[int, dict[torch.device, torch.Tensor]]] = None,
):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model to make sure they have the right
    execution device

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`int`, `str` or `torch.device`):
            The device on which inputs and model weights should be placed before the forward pass.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    if not hasattr(module, "_hf_hook") and len(module.state_dict()) > 0:
        add_hook_to_module(
            module,
            AlignDevicesHook(execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map),
        )

    # Break the recursion if we get to a preload module.
    if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
        return

    for child in module.children():
        attach_execution_device_hook(
            child,
            execution_device,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )


def attach_align_device_hook(
    module: torch.nn.Module,
    execution_device: Optional[torch.device] = None,
    offload: bool = False,
    weights_map: Optional[Mapping] = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys: Optional[Union[str, list[str]]] = None,
    preload_module_classes: Optional[list[str]] = None,
    tied_params_map: Optional[dict[int, dict[torch.device, torch.Tensor]]] = None,
):
    """
    Recursively attaches `AlignDevicesHook` to all submodules of a given model that have direct parameters and/or
    buffers.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    # Attach the hook on this module if it has any direct tensor.
    directs = named_module_tensors(module)
    full_offload = (
        offload and preload_module_classes is not None and module.__class__.__name__ in preload_module_classes
    )

    if len(list(directs)) > 0 or full_offload:
        if weights_map is not None:
            prefix = f"{module_name}." if len(module_name) > 0 else ""
            prefixed_weights_map = PrefixedDataset(weights_map, prefix)
        else:
            prefixed_weights_map = None
        hook = AlignDevicesHook(
            execution_device=execution_device,
            offload=offload,
            weights_map=prefixed_weights_map,
            offload_buffers=offload_buffers,
            place_submodules=full_offload,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
        add_hook_to_module(module, hook, append=True)

    # We stop the recursion in case we hit the full offload.
    if full_offload:
        return

    # Recurse on all children of the module.
    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
        attach_align_device_hook(
            child,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=child_name,
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )


def remove_hook_from_submodules(module: nn.Module):
    """
    Recursively removes all hooks attached on the submodules of a given model.

    Args:
        module (`torch.nn.Module`): The module on which to remove all hooks.
    """
    remove_hook_from_module(module)
    for child in module.children():
        remove_hook_from_submodules(child)


def attach_align_device_hook_on_blocks(
    module: nn.Module,
    execution_device: Optional[Union[torch.device, dict[str, torch.device]]] = None,
    offload: Union[bool, dict[str, bool]] = False,
    weights_map: Mapping = None,
    offload_buffers: bool = False,
    module_name: str = "",
    skip_keys: Optional[Union[str, list[str]]] = None,
    preload_module_classes: Optional[list[str]] = None,
    tied_params_map: Optional[dict[int, dict[torch.device, torch.Tensor]]] = None,
):
    """
    Attaches `AlignDevicesHook` to all blocks of a given model as needed.

    Args:
        module (`torch.nn.Module`):
            The module where we want to attach the hooks.
        execution_device (`torch.device` or `Dict[str, torch.device]`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass. It can be one device
            for the whole module, or a dictionary mapping module name to device.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass. It can be one boolean for the whole
            module, or a dictionary mapping module name to boolean.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        module_name (`str`, *optional*, defaults to `""`):
            The name of the module.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        tied_params_map (Optional[Dict[int, Dict[torch.device, torch.Tensor]]], *optional*, defaults to `None`):
            A map of data pointers to dictionaries of devices to already dispatched tied weights. For a given execution
            device, this parameter is useful to reuse the first available pointer of a shared weight for all others,
            instead of duplicating memory.
    """
    # If one device and one offload, we've got one hook.
    if not isinstance(execution_device, Mapping) and not isinstance(offload, dict):
        if not offload:
            hook = AlignDevicesHook(
                execution_device=execution_device,
                io_same_device=True,
                skip_keys=skip_keys,
                place_submodules=True,
                tied_params_map=tied_params_map,
            )
            add_hook_to_module(module, hook)
        else:
            attach_align_device_hook(
                module,
                execution_device=execution_device,
                offload=True,
                weights_map=weights_map,
                offload_buffers=offload_buffers,
                module_name=module_name,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
        return

    if not isinstance(execution_device, Mapping):
        execution_device = {key: execution_device for key in offload.keys()}
    if not isinstance(offload, Mapping):
        offload = {key: offload for key in execution_device.keys()}

    if module_name in execution_device and module_name in offload and not offload[module_name]:
        hook = AlignDevicesHook(
            execution_device=execution_device[module_name],
            offload_buffers=offload_buffers,
            io_same_device=(module_name == ""),
            place_submodules=True,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
        add_hook_to_module(module, hook)
        attach_execution_device_hook(
            module, execution_device[module_name], skip_keys=skip_keys, tied_params_map=tied_params_map
        )
    elif module_name in execution_device and module_name in offload:
        attach_align_device_hook(
            module,
            execution_device=execution_device[module_name],
            offload=True,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=module_name,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )
        if not hasattr(module, "_hf_hook"):
            hook = AlignDevicesHook(
                execution_device=execution_device[module_name],
                io_same_device=(module_name == ""),
                skip_keys=skip_keys,
                tied_params_map=tied_params_map,
            )
            add_hook_to_module(module, hook)
        attach_execution_device_hook(
            module,
            execution_device[module_name],
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
    elif module_name == "":
        hook = AlignDevicesHook(
            execution_device=execution_device.get(""),
            io_same_device=True,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )
        add_hook_to_module(module, hook)

    for child_name, child in module.named_children():
        child_name = f"{module_name}.{child_name}" if len(module_name) > 0 else child_name
        attach_align_device_hook_on_blocks(
            child,
            execution_device=execution_device,
            offload=offload,
            weights_map=weights_map,
            offload_buffers=offload_buffers,
            module_name=child_name,
            preload_module_classes=preload_module_classes,
            skip_keys=skip_keys,
            tied_params_map=tied_params_map,
        )


class CpuOffload(ModelHook):
    """
    Offloads a model on the CPU until its forward pass is called. The model will not be offloaded back to the CPU after
    the forward, the user needs to call the `init_hook` method again for this.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by [`cpu_offload_with_hook`] for a previous model in the pipeline you are running. If
            passed, its offload method will be called just before the forward of the model to which this hook is
            attached.
    """

    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        prev_module_hook: Optional["UserCpuOffloadHook"] = None,
    ):
        self.prev_module_hook = prev_module_hook

        self.execution_device = execution_device if execution_device is not None else PartialState().default_device

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None and isinstance(self.prev_module_hook, UserCpuOffloadHook):
            prev_module = self.prev_module_hook.model
            prev_device = next(prev_module.parameters()).device

            # Only offload the previous module if it is not already on CPU.
            if prev_device != torch.device("cpu"):
                self.prev_module_hook.offload()
                clear_device_cache()

        # If the current device is already the self.execution_device, we can skip the transfer.
        current_device = next(module.parameters()).device
        if current_device == self.execution_device:
            return args, kwargs

        module.to(self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class UserCpuOffloadHook:
    """
    A simple hook grouping a model and a `ModelHook`, which provides easy APIs for to call the init method of the hook
    or remove it entirely.
    """

    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)


class LayerwiseCastingHook(ModelHook):
    r"""
    A hook that casts the weights of a module to a high precision dtype for computation, and to a low precision dtype
    for storage. This process may lead to quality loss in the output, but can significantly reduce the memory
    footprint.
    """

    _is_stateful = False

    def __init__(self, storage_dtype: torch.dtype, compute_dtype: torch.dtype, non_blocking: bool) -> None:
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.non_blocking = non_blocking

    def init_hook(self, module: torch.nn.Module):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return module

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        module.to(dtype=self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output):
        module.to(dtype=self.storage_dtype, non_blocking=self.non_blocking)
        return output
