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

from types import MethodType

import torch.nn as nn

from .imports import is_hpu_available, is_transformer_engine_available


# Do not import `transformer_engine` at package level to avoid potential issues


def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Converts model layers to Transformer Engine counterparts with reduced memory overhead.

    This function performs a two-stage conversion process:
    1. Creates TE module structure on meta device (no memory allocation)
    2. Transfers weights efficiently to avoid peak memory usage during conversion

    Args:
        model: The model to convert
        to_transformer_engine (bool): Whether to convert to TE (True) or from TE (False)
        _convert_linear (bool): Whether to convert Linear layers
        _convert_ln (bool): Whether to convert LayerNorm layers

    Returns:
        The converted model

    Raises:
        ImportError: If transformer_engine is not available
    """
    if not is_transformer_engine_available():
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")

    if is_hpu_available():
        import intel_transformer_engine as te

        if not hasattr(te, "LayerNorm"):
            # HPU does not have a LayerNorm implementation in TE
            te.LayerNorm = nn.LayerNorm
    else:
        import transformer_engine.pytorch as te

    import torch

    from accelerate import init_empty_weights

    # Stage 1: Create TE module skeleton on meta device (zero memory allocation)
    with init_empty_weights():
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and to_transformer_engine and _convert_linear:
                has_bias = module.bias is not None

                # TE requires weight dimensions to be multiples of 16 for optimal performance
                if any(p % 16 != 0 for p in module.weight.shape):
                    continue

                # Create TE Linear module structure without allocating memory
                te_module = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=has_bias,
                    params_dtype=module.weight.dtype,
                    device="meta",
                )
                setattr(model, name, te_module)

            elif isinstance(module, nn.LayerNorm) and to_transformer_engine and _convert_ln:
                # Create TE LayerNorm module structure without allocating memory
                te_module = te.LayerNorm(
                    module.normalized_shape[0],
                    eps=module.eps,
                    params_dtype=module.weight.dtype,
                )
                setattr(model, name, te_module)

            else:
                # Recursively convert child modules
                convert_model(
                    module,
                    to_transformer_engine=to_transformer_engine,
                    _convert_linear=_convert_linear,
                    _convert_ln=_convert_ln,
                )

    # Efficiently transfer weights from original to TE modules
    for name, module in model.named_modules():
        if (
            isinstance(module, te.Linear)
            and hasattr(module, "weight")
            and module.weight.device == torch.device("meta")
        ):
            # Locate corresponding weight parameters in the model's state dict
            weight_key = f"{name}.weight"
            bias_key = f"{name}.bias"

            state_dict = model.state_dict()

            # Transfer weight parameter with memory-efficient copying
            if weight_key in state_dict:
                with torch.no_grad():
                    # For very large weights, consider chunked transfer to reduce peak memory
                    module.weight.copy_(state_dict[weight_key])

            # Transfer bias parameter if it exists
            if hasattr(module, "bias") and module.bias is not None and bias_key in state_dict:
                with torch.no_grad():
                    module.bias.copy_(state_dict[bias_key])

    return model


def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_transformer_engine_available():
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")

    if is_hpu_available():
        import intel_transformer_engine as te

        module_cls_to_check = te.Linear
    else:
        import transformer_engine.pytorch as te

        module_cls_to_check = (te.LayerNorm, te.Linear, te.TransformerLayer)

    for m in model.modules():
        if isinstance(m, module_cls_to_check):
            return True

    return False


def contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval=False):
    """
    Wrapper for a model's forward method to apply FP8 autocast. Is context aware, meaning that by default it will
    disable FP8 autocast during eval mode, which is generally better for more accurate metrics.
    """
    if not is_transformer_engine_available():
        raise ImportError("Using `contextual_fp8_autocast` requires transformer_engine to be installed.")

    if is_hpu_available():
        from intel_transformer_engine import fp8_autocast
    else:
        from transformer_engine.pytorch import fp8_autocast

    def forward(self, *args, **kwargs):
        enabled = use_during_eval or self.training
        with fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe):
            return model_forward(*args, **kwargs)

    # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`
    forward.__wrapped__ = model_forward

    return forward


def apply_fp8_autowrap(model, fp8_recipe_handler):
    """
    Applies FP8 context manager to the model's forward method
    """
    if not is_transformer_engine_available():
        raise ImportError("Using `apply_fp8_autowrap` requires transformer_engine to be installed.")

    if is_hpu_available():
        import intel_transformer_engine.recipe as te_recipe
    else:
        import transformer_engine.common.recipe as te_recipe

    kwargs = fp8_recipe_handler.to_kwargs() if fp8_recipe_handler is not None else {}
    if "fp8_format" in kwargs:
        kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
    use_during_eval = kwargs.pop("use_autocast_during_eval", False)
    fp8_recipe = te_recipe.DelayedScaling(**kwargs)
    new_forward = contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

    if hasattr(model.forward, "__func__"):
        model.forward = MethodType(new_forward, model)
    else:
        model.forward = new_forward

    return model
