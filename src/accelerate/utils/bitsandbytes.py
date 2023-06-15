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


import logging

import torch
import torch.nn as nn

from .imports import (
    is_4bit_bnb_available,
    is_8bit_bnb_available,
    is_bnb_available,
    is_peft_available,
    is_transformers_available,
)


if is_bnb_available():
    import bitsandbytes as bnb

logger = logging.getLogger(__name__)


def get_bnb_model(model, bnb_config, is_peft_model):
    """
        This function will quantize the input model with the parameters specified in bnb_config.

    Args:
        model (`torch.nn.Module`): Input model
        bnb_config (`dict`'): The bitsandbytes parameters

    Returns:
        `torch.nn.Module`: The quantized model
    """
    if is_transformers_available():
        from transformers import BitsAndBytesConfig
        from transformers.utils.bitsandbytes import get_keys_to_not_convert

    quantization_config = BitsAndBytesConfig.from_dict(bnb_config, return_unused_kwargs=False)

    load_in_4bit = quantization_config.load_in_4bit
    load_in_8bit = quantization_config.load_in_8bit

    if (load_in_4bit and not is_4bit_bnb_available()) or (load_in_8bit and not is_8bit_bnb_available()):
        raise ImportError(
            "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training."
            "Make sure you have the latest version of `bitsandbytes` installed"
        )
    # get dtype
    # set dtype of the model disregarding what we have in config
    torch_dtype = model.dtype
    if torch_dtype != torch.float16:
        # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
        logger.info(
            f"Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to "
            "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
            "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
            " torch_dtype=torch.float16 to remove this warning."
        )
        torch_dtype = torch.float16
        model.to(torch_dtype)

    # some modules needs to be in fp32
    keep_in_fp32_modules = model._keep_in_fp32_modules
    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                param = param.to(torch.float32)

    # set model config
    config = model.config
    if load_in_8bit and quantization_config is not None:
        if hasattr(config, "quantization_config"):
            logger.warning(
                " The model you're loading already has a `quantization_config` attribute."
                "The `quantization_config` attribute will be overwritten with the"
                " one you have set up using `accelerate config`"
            )
        config.quantization_config = quantization_config

    # if we need to skip modules
    llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    if len(llm_int8_skip_modules) == 0:
        if is_peft_model:
            modules_to_not_convert = get_keys_to_not_convert(model.base_model.model)
        else:
            modules_to_not_convert = get_keys_to_not_convert(model)
    else:
        modules_to_not_convert = quantization_config.llm_int8_skip_modules
    if not isinstance(modules_to_not_convert, list):
        modules_to_not_convert = [modules_to_not_convert]
    if keep_in_fp32_modules is not None:
        modules_to_not_convert.extend(keep_in_fp32_modules)

    if load_in_4bit:
        if not is_4bit_bnb_available():
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )
    model = replace_with_bnb_linear_with_weight(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
        is_peft_model=is_peft_model,
    )

    model.config.quantization_config = quantization_config
    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit
    model.is_quantized = load_in_8bit or load_in_4bit
    model._is_quantized_training_enabled = True

    return model


def replace_with_bnb_linear_with_weight(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, is_peft_model=None
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules or by `bnb.nn.Linear4bit`
    modules from the `bitsandbytes`library. If the model class is `PeftModel`, we replace the `torch.nn.Linear` modules
    specified in `peft_config` by `peft.tuners.lora.Linear8bit` modules or by `peft.tuners.lora.Linear4bit` modules.
    The function will be run recursively and replace `torch.nn.Linear` modules except for the `lm_head` that should be
    kept as a `torch.nn.Linear` module.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert.
    """

    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    if is_peft_model:
        peft_config = model.peft_config
    else:
        peft_config = None

    model, has_been_replaced = _replace_with_bnb_linear_with_weight(
        model, modules_to_not_convert, current_key_name, quantization_config, peft_config
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model


def _replace_with_bnb_linear_with_weight(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, peft_config=None
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    if peft_config:
        if is_peft_available():
            import peft.tuners.lora as lora

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                if quantization_config.load_in_8bit:
                    peft_module_found = False
                    if peft_config is not None:
                        for adapter_name, lora_config in peft_config.items():
                            if name in lora_config.target_modules:
                                lora_kwargs = {
                                    "r": lora_config.r,
                                    "lora_alpha": lora_config.lora_alpha,
                                    "lora_dropout": lora_config.lora_dropout,
                                    "fan_in_fan_out": lora_config.fan_in_fan_out,
                                    "init_lora_weights": lora_config.init_lora_weights,
                                }
                                eightbit_kwargs = lora_kwargs.copy()
                                eightbit_kwargs.update(
                                    {
                                        "has_fp16_weights": quantization_config.llm_int8_has_fp16_weight,
                                        "threshold": quantization_config.llm_int8_threshold,
                                    }
                                )
                                bnb_module = lora.Linear8bitLt(
                                    adapter_name,
                                    module.in_features,
                                    module.out_features,
                                    bias=module.bias is not None,
                                    **eightbit_kwargs,
                                )
                                peft_module_found = True
                                break
                    if not peft_module_found:
                        bnb_module = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                    bnb_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        bnb_module.bias.data = module.bias.data.clone()
                    setattr(model, name, bnb_module)
                    has_been_replaced = True
                elif quantization_config.load_in_4bit:
                    peft_module_found = False
                    if peft_config is not None:
                        for adapter_name, lora_config in peft_config.items():
                            if name in lora_config.target_modules:
                                lora_kwargs = {
                                    "r": lora_config.r,
                                    "lora_alpha": lora_config.lora_alpha,
                                    "lora_dropout": lora_config.lora_dropout,
                                    "fan_in_fan_out": lora_config.fan_in_fan_out,
                                    "init_lora_weights": lora_config.init_lora_weights,
                                }
                                fourbit_kwargs = lora_kwargs.copy()
                                fourbit_kwargs.update(
                                    {
                                        "compute_dtype": quantization_config.bnb_4bit_compute_dtype,
                                        "compress_statistics": quantization_config.bnb_4bit_use_double_quant,
                                        "quant_type": quantization_config.bnb_4bit_quant_type,
                                    }
                                )
                                bnb_module = lora.Linear4bit(
                                    adapter_name,
                                    module.in_features,
                                    module.out_features,
                                    bias=module.bias is not None,
                                    **fourbit_kwargs,
                                )
                                peft_module_found = True
                                break
                    if not peft_module_found:
                        bnb_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                        )
                    bnb_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        bnb_module.bias.data = module.bias.data.clone()
                    setattr(model, name, bnb_module)
                    has_been_replaced = True
                # Force requires grad to False to avoid unexpected errors
                module.requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear_with_weight(
                module, modules_to_not_convert, current_key_name, quantization_config, peft_config
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def has_bnb_layers(model):
    """Check if we have `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt` layers inside our model"""
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit) or isinstance(m, bnb.nn.Linear8bitLt):
            return True
    return False


def prepare_model_for_kbit_peft_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32 4- make sure to not modfiy lora layers

    Args:
        model, (`nn.torch.Module`):
            The input model
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        if "lora_" not in name:
            # freeze base model's layers
            param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if "lora_" not in name:
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    model.train()
    return model
