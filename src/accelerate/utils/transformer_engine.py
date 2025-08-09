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
from .operations import GatheredParameters


# Do not import `transformer_engine` at package level to avoid potential issues


def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
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

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            params_to_gather = [module.weight]
            if has_bias:
                params_to_gather.append(module.bias)

            with GatheredParameters(params_to_gather, modifier_rank=0):
                if any(p % 16 != 0 for p in module.weight.shape):
                    return
                te_module = te.Linear(
                    module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
                )
                te_module.weight.copy_(module.weight)
                if has_bias:
                    te_module.bias.copy_(module.bias)

                setattr(model, name, te_module)
        # Note: @xrsrke (Phuc) found that te.LayerNorm doesn't have any real memory savings or speedups over nn.LayerNorm
        elif isinstance(module, nn.LayerNorm) and to_transformer_engine and _convert_ln:
            with GatheredParameters([module.weight, module.bias], modifier_rank=0):
                te_module = te.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
                te_module.weight.copy_(module.weight)
                te_module.bias.copy_(module.bias)

            setattr(model, name, te_module)
        elif isinstance(module, te.Linear) and not to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            new_module = nn.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        elif isinstance(module, te.LayerNorm) and not to_transformer_engine and _convert_ln:
            new_module = nn.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            new_module.weight.copy_(module.weight)
            new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        else:
            convert_model(
                module,
                to_transformer_engine=to_transformer_engine,
                _convert_linear=_convert_linear,
                _convert_ln=_convert_ln,
            )


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
