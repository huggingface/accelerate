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
import torch
import torch.nn as nn

from .imports import is_fp8_available


if is_fp8_available():
    import transformer_engine.pytorch as te


def get_nested_children(model: nn.Module):
    """
    Finds all nested children in `model` and returns it as a nested dict of `layer_name: layer`.
    """
    children = dict(model.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = get_nested_children(child)
            except TypeError:
                output[name] = get_nested_children(child)
    return output


def check_for_layer(dictionary, layer, current_key=None):
    """
    Checks for locations of `layer` in `dictionary` and returns a list of keys to those locations.

    Locations are in the format of `["a", "0", "c"]` to mean `a[0].c`.
    """
    if current_key is None:
        current_key = []
    matching_keys = []

    for k, v in dictionary.items():
        if isinstance(v, layer):
            matching_keys.append(".".join(current_key + [k]))
        elif isinstance(v, dict):
            matching_keys.extend(check_for_layer(v, layer, current_key + [k]))

    return matching_keys


def _replace_layers(model, keys: list, replacement_function: callable):
    """
    Finds layers based on `keys` inside `model` and replaces those layers with `replacement`.

    Keys should be in the format of `["a", "0", "c"]` to mean `model.a[0].c`.

    Args:
        model (nn.Module):
            model to replace layers in
        keys (list):
            list of keys to find the layer to replace
        replacement_function (callable):
            function to replace the layer with. Should return a layer.
    """
    current_module = model

    for key in keys[:-1]:
        if isinstance(key, int):
            current_module = current_module[key]
        elif isinstance(current_module, nn.Module) and key in current_module._modules:
            current_module = current_module._modules[key]
        elif hasattr(current_module, key):
            current_module = getattr(current_module, key)
        else:
            raise KeyError(f"Key '{key}' not found in the model.")

    last_key = keys[-1]
    if isinstance(last_key, int):
        current_module[last_key] = replacement_function(current_module[last_key])
    elif isinstance(current_module, nn.Module) and last_key in current_module._modules:
        current_module._modules[last_key] = replacement_function(current_module._modules[last_key])
    elif hasattr(current_module, last_key):
        setattr(current_module, last_key, replacement_function(getattr(current_module, last_key)))
    else:
        raise KeyError(f"Key '{last_key}' not found in the model.")

    return model


def convert_linear(model, to_fp8=True):
    """
    Converts all linear layers in `model` to `transformer_engine` linear layers. If `to_fp8` is False, the layers are
    converted back to `nn.Linear`.
    """
    locations = check_for_layer(get_nested_children(model), nn.Linear if to_fp8 else te.Linear)
    new_layer = te.Linear if to_fp8 else nn.Linear

    def _inner(module):
        # Return early if the linear layer weights are not multiples of 16
        if any(p % 16 != 0 for p in module.weight.shape):
            return module
        has_bias = module.bias is not None
        new_module = new_layer(
            module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
        )
        module.weight.copy_(new_module.weight)
        if has_bias:
            module.bias.copy_(new_module.bias)
        return new_module

    with torch.no_grad():
        for location in locations:
            model = _replace_layers(model, location.split("."), replacement_function=_inner)
    return model


def convert_layernorm(model, to_fp8=True):
    """
    Converts all layernorm layers in `model` to `transformer_engine` layernorm layers. If `to_fp8` is False, the layers
    are converted back to `nn.LayerNorm`.
    """
    locations = check_for_layer(get_nested_children(model), nn.LayerNorm if to_fp8 else te.LayerNorm)
    new_layer = te.LayerNorm if to_fp8 else nn.LayerNorm

    def _inner(module):
        new_module = new_layer(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
        module.weight.copy_(new_module.weight)
        module.bias.copy_(new_module.bias)
        return new_module

    with torch.no_grad():
        for location in locations:
            model = _replace_layers(model, location.split("."), replacement_function=_inner)
    return model


def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if not is_fp8_available():
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")
    if to_transformer_engine:
        if _convert_linear:
            model = convert_linear(model)
        if _convert_ln:
            model = convert_layernorm(model)
    else:
        if _convert_linear:
            model = convert_linear(model, to_fp8=False)
        if _convert_ln:
            model = convert_layernorm(model, to_fp8=False)


def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_fp8_available():
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")
    children = get_nested_children(model)
    if len(check_for_layer(children, te.Linear)) > 0 or len(check_for_layer(children, te.LayerNorm)) > 0:
        return True
    return False
