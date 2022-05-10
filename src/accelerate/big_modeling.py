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
from contextlib import contextmanager
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .hooks import AlignDevicesHook, add_hook_to_module, attach_align_device_hook, attach_align_device_hook_on_blocks
from .utils import OffloadedWeightsLoader, check_device_map, extract_submodules_state_dict, offload_state_dict


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```pyton
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            module._parameters[name] = nn.Parameter(module._parameters[name].to(torch.device("meta")))

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


def cpu_offload(
    model: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Activates full CPU offload for a model. As a result, all parameters of the model will be offloaded and only one
    copy of the state dict of the model will be kept. During the forward pass, parameters will be extracted from that
    state dict and put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the model that will be kept on CPU.
    """
    if execution_device is None:
        execution_device = next(iter(model.parameter())).device
    if state_dict is None:
        state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}
    attach_align_device_hook(
        model, execution_device=execution_device, offload=True, offload_buffers=offload_buffers, weights_map=state_dict
    )
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True))
    return model


def disk_offload(
    model: nn.Module,
    offload_dir: Union[str, os.PathLike],
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
):
    """
    Activates full disk offload for a model. As a result, all parameters of the model will be offloaded as
    memory-mapped array in a given folder. During the forward pass, parameters will be accessed from that folder and
    put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`): The model to offload.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model's first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
    """
    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameter())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
    )
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True))
    return model


def dispatch_model(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Union[str, os.PathLike] = None,
    offload_buffers: bool = False,
):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
    """
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    if main_device is None:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_modules = [name for name, device in device_map.items() if device == "cpu"]
    if state_dict is None and len(cpu_modules) > 0:
        state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

    disk_modules = [name for name, device in device_map.items() if device == "disk"]
    if offload_dir is None and len(disk_modules) > 0:
        raise ValueError(
            "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
            f"need to be offloaded: {', '.join(disk_modules)}."
        )
    if len(disk_modules) > 0 and (
        not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json"))
    ):
        disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
        offload_state_dict(offload_dir, disk_state_dict)

    execution_device = {
        name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
    }
    offload = {name: device in ["cpu", "disk"] for name, device in device_map.items()}
    if state_dict is not None or offload_dir is not None:
        weights_map = OffloadedWeightsLoader(state_dict=state_dict, save_folder=offload_dir)
    else:
        weights_map = None

    attach_align_device_hook_on_blocks(
        model,
        execution_device=execution_device,
        offload=offload,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
    )
    return model
