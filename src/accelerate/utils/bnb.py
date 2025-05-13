# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import logging
import os
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn

from accelerate.utils.imports import (
    is_4bit_bnb_available,
    is_8bit_bnb_available,
)

from ..big_modeling import dispatch_model, init_empty_weights
from .dataclasses import BnbQuantizationConfig
from .modeling import (
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    load_checkpoint_in_model,
    offload_weight,
    set_module_tensor_to_device,
)


logger = logging.getLogger(__name__)


def load_and_quantize_model(
    model: torch.nn.Module,
    bnb_quantization_config: BnbQuantizationConfig,
    weights_location: Union[str, os.PathLike] = None,
    device_map: Optional[dict[str, Union[int, str, torch.device]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_state_dict: bool = False,
):
    """
    This function will quantize the input model with the associated config passed in `bnb_quantization_config`. If the
    model is in the meta device, we will load and dispatch the weights according to the `device_map` passed. If the
    model is already loaded, we will quantize the model and put the model on the GPU,

    Args:
        model (`torch.nn.Module`):
            Input model. The model can be already loaded or on the meta device
        bnb_quantization_config (`BnbQuantizationConfig`):
            The bitsandbytes quantization parameters
        weights_location (`str` or `os.PathLike`):
            The folder weights_location to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
            - a path to a folder containing a unique pytorch_model.bin file.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.

    Returns:
        `torch.nn.Module`: The quantized model
    """

    load_in_4bit = bnb_quantization_config.load_in_4bit
    load_in_8bit = bnb_quantization_config.load_in_8bit

    if load_in_8bit and not is_8bit_bnb_available():
        raise ImportError(
            "You have a version of `bitsandbytes` that is not compatible with 8bit quantization,"
            " make sure you have the latest version of `bitsandbytes` installed."
        )
    if load_in_4bit and not is_4bit_bnb_available():
        raise ValueError(
            "You have a version of `bitsandbytes` that is not compatible with 4bit quantization,"
            "make sure you have the latest version of `bitsandbytes` installed."
        )

    modules_on_cpu = []
    # custom device map
    if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        modules_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    if bnb_quantization_config.skip_modules is None:
        bnb_quantization_config.skip_modules = get_keys_to_not_convert(model)

    # add cpu modules to skip modules only for 4-bit modules
    if load_in_4bit:
        bnb_quantization_config.skip_modules.extend(modules_on_cpu)
    modules_to_not_convert = bnb_quantization_config.skip_modules

    # We add the modules we want to keep in full precision
    if bnb_quantization_config.keep_in_fp32_modules is None:
        bnb_quantization_config.keep_in_fp32_modules = []
    keep_in_fp32_modules = bnb_quantization_config.keep_in_fp32_modules
    modules_to_not_convert.extend(keep_in_fp32_modules)

    # compatibility with peft
    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit

    model_device = get_parameter_device(model)
    if model_device.type != "meta":
        # quantization of an already loaded model
        logger.warning(
            "It is not recommended to quantize a loaded model. "
            "The model should be instantiated under the `init_empty_weights` context manager."
        )
        model = replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert)
        # convert param to the right dtype
        dtype = bnb_quantization_config.torch_dtype
        for name, param in model.state_dict().items():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                param.to(torch.float32)
                if param.dtype != torch.float32:
                    name = name.replace(".weight", "").replace(".bias", "")
                    param = getattr(model, name, None)
                    if param is not None:
                        param.to(torch.float32)
            elif torch.is_floating_point(param):
                param.to(dtype)
        if model_device.type == "cuda":
            model.cuda(torch.cuda.current_device())
            torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            model.to(torch.cuda.current_device())
        elif torch.xpu.is_available():
            model.to(torch.xpu.current_device())
        else:
            raise RuntimeError("No GPU or Intel XPU found. A GPU or Intel XPU is needed for quantization.")
        logger.info(
            f"The model device type is {model_device.type}. However, gpu or intel xpu is needed for quantization."
            "We move the model to it."
        )
        return model

    elif weights_location is None:
        raise RuntimeError(
            f"`weights_location` needs to be the folder path containing the weights of the model, but we found {weights_location} "
        )

    else:
        with init_empty_weights():
            model = replace_with_bnb_layers(
                model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert
            )
        device_map = get_quantized_model_device_map(
            model,
            bnb_quantization_config,
            device_map,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
            offload_state_dict = True

        offload = any(x in list(device_map.values()) for x in ["cpu", "disk"])

        load_checkpoint_in_model(
            model,
            weights_location,
            device_map,
            dtype=bnb_quantization_config.torch_dtype,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            keep_in_fp32_modules=bnb_quantization_config.keep_in_fp32_modules,
            offload_8bit_bnb=load_in_8bit and offload,
        )
        return dispatch_model(model, device_map=device_map, offload_dir=offload_folder)


def get_quantized_model_device_map(
    model, bnb_quantization_config, device_map=None, max_memory=None, no_split_module_classes=None
):
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif torch.xpu.is_available():
            device_map = {"": torch.xpu.current_device()}
        else:
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
        logger.info("The device_map was not initialized.Setting device_map to `{'':torch.cuda.current_device()}`.")

    if isinstance(device_map, str):
        if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )

        special_dtypes = {}
        special_dtypes.update(
            {
                name: bnb_quantization_config.torch_dtype
                for name, _ in model.named_parameters()
                if any(m in name for m in bnb_quantization_config.skip_modules)
            }
        )
        special_dtypes.update(
            {
                name: torch.float32
                for name, _ in model.named_parameters()
                if any(m in name for m in bnb_quantization_config.keep_in_fp32_modules)
            }
        )

        kwargs = {}
        kwargs["special_dtypes"] = special_dtypes
        kwargs["no_split_module_classes"] = no_split_module_classes
        kwargs["dtype"] = bnb_quantization_config.target_dtype

        # get max_memory for each device.
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **kwargs,
            )

        kwargs["max_memory"] = max_memory
        device_map = infer_auto_device_map(model, **kwargs)

    if isinstance(device_map, dict):
        # check if don't have any quantized module on the cpu
        modules_not_to_convert = bnb_quantization_config.skip_modules + bnb_quantization_config.keep_in_fp32_modules

        device_map_without_some_modules = {
            key: device_map[key] for key in device_map.keys() if key not in modules_not_to_convert
        }
        for device in ["cpu", "disk"]:
            if device in device_map_without_some_modules.values():
                if bnb_quantization_config.load_in_4bit:
                    raise ValueError(
                        """
                        Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
                        the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
                        these modules in `torch_dtype`, you need to pass a custom `device_map` to
                        `load_and_quantize_model`. Check
                        https://huggingface.co/docs/accelerate/main/en/usage_guides/quantization#offload-modules-to-cpu-and-disk
                        for more details.
                        """
                    )
                else:
                    logger.info(
                        "Some modules are are offloaded to the CPU or the disk. Note that these modules will be converted to 8-bit"
                    )
        del device_map_without_some_modules
    return device_map


def replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=None, current_key_name=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules or by `bnb.nn.Linear4bit`
    modules from the `bitsandbytes`library. The function will be run recursively and replace `torch.nn.Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[str]`):
            Names of the modules to not quantize convert. In practice we keep the `lm_head` in full precision for
            numerical stability reasons.
        current_key_name (`List[str]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert.
    """

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    model, has_been_replaced = _replace_with_bnb_layers(
        model, bnb_quantization_config, modules_to_not_convert, current_key_name
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model


def _replace_with_bnb_layers(
    model,
    bnb_quantization_config,
    modules_to_not_convert=None,
    current_key_name=None,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    # bitsandbytes will initialize CUDA on import, so it needs to be imported lazily
    import bitsandbytes as bnb

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            proceed = True
            for key in modules_to_not_convert:
                if (
                    (key in current_key_name_str) and (key + "." in current_key_name_str)
                ) or key == current_key_name_str:
                    proceed = False
                    break
            if proceed:
                # Load bnb module with empty weight and replace ``nn.Linear` module
                if bnb_quantization_config.load_in_8bit:
                    bnb_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=bnb_quantization_config.llm_int8_threshold,
                    )
                elif bnb_quantization_config.load_in_4bit:
                    bnb_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        bnb_quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=bnb_quantization_config.bnb_4bit_use_double_quant,
                        quant_type=bnb_quantization_config.bnb_4bit_quant_type,
                    )
                else:
                    raise ValueError("load_in_8bit and load_in_4bit can't be both False")
                bnb_module.weight.data = module.weight.data
                if module.bias is not None:
                    bnb_module.bias.data = module.bias.data
                bnb_module.requires_grad_(False)
                setattr(model, name, bnb_module)
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = _replace_with_bnb_layers(
                module, bnb_quantization_config, modules_to_not_convert, current_key_name
            )
            has_been_replaced = has_been_replaced | _has_been_replaced
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model
    with init_empty_weights():
        tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = False
    if hasattr(model, "base_model_prefix"):
        is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_children())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


def has_4bit_bnb_layers(model):
    """Check if we have `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt` layers inside our model"""
    # bitsandbytes will initialize CUDA on import, so it needs to be imported lazily
    import bitsandbytes as bnb

    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            return True
    return False


def get_parameter_device(parameter: nn.Module):
    return next(parameter.parameters()).device


def quantize_and_offload_8bit(model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics):
    # if it is not quantized, we quantize and offload the quantized weights and the SCB stats
    if fp16_statistics is None:
        set_module_tensor_to_device(model, param_name, 0, dtype=new_dtype, value=param)
        tensor_name = param_name
        module = model
        if "." in tensor_name:
            splits = tensor_name.split(".")
            for split in splits[:-1]:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f"{module} has no attribute {split}.")
                module = new_module
            tensor_name = splits[-1]
        # offload weights
        module._parameters[tensor_name].requires_grad = False
        offload_weight(module._parameters[tensor_name], param_name, offload_folder, index=offload_index)
        if hasattr(module._parameters[tensor_name], "SCB"):
            offload_weight(
                module._parameters[tensor_name].SCB,
                param_name.replace("weight", "SCB"),
                offload_folder,
                index=offload_index,
            )
    else:
        offload_weight(param, param_name, offload_folder, index=offload_index)
        offload_weight(fp16_statistics, param_name.replace("weight", "SCB"), offload_folder, index=offload_index)

    set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype, value=torch.empty(*param.size()))
