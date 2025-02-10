# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Needed utilities for torchao FP8 training.
"""

from functools import partial

import torch

from .imports import torchao_required


def find_first_last_linear_layers(model: torch.nn.Module):
    """
    Finds the first and last linear layer names in a model.

    This is needed during FP8 to avoid issues with instability by keeping the first and last layers unquantized.

    Ref: https://x.com/xariusrke/status/1826669142604141052
    """
    first_linear, last_linear = None, None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if first_linear is None:
                first_linear = name
            last_linear = name
    return first_linear, last_linear


def filter_linear_layers(module, layer_name, first_layer_name, last_layer_name) -> bool:
    """
    A function which will check if `module` is:
    - a `torch.nn.Linear` layer
    - has in_features and out_features divisible by 16
    - is not the first or last layer of the model.

    Args:
        module (`torch.nn.Module`):
            The module to check.
        layer_name (`str`):
            The fully qualified name of the layer.
        first_layer_name (`str`):
            The name of the first layer of the model.
        last_layer_name (`str`):
            The name of the last layer of the model.
    """
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    # TODO: apply this to all FP8 backends
    if layer_name in (first_layer_name, last_layer_name):
        return False
    return True


@torchao_required
def has_ao_layers(model: torch.nn.Module):
    from torchao.float8.float8_linear import Float8Linear

    for name, module in model.named_modules():
        if isinstance(module, Float8Linear):
            return True
    return False


@torchao_required
def convert_to_float8_training(
    model: torch.nn.Module,
    config=None,
    module_filter_func=None,
):
    """
    Converts all `nn.Linear` layers in the model (except the first and last) to torchao's `Float8Linear` layer inplace.

    Args:
        model (`torch.nn.Module`):
            The model to convert.
        config (`torchao.float8.Float8LinearConfig`, *optional*):
            The configuration for the FP8 training. Recommended to utilize
            `torchao.float8.recipe_name_to_linear_config` to generate this. In general, the default config should be
            sufficient.
        module_filter_func (`Callable`, *optional*):
            Optional function that must take in a module and layer name, and returns a boolean indicating whether the
            module should be converted to FP8. Defaults to `filter_linear_layers`. See it for an example.

    Example:

    ```python
    from accelerate.utils.ao import convert_to_float8_training

    model = MyModel()
    model.to("cuda")
    convert_to_float8_training(model)

    model.train()
    ```
    """
    from torchao.float8 import convert_to_float8_training

    first_linear, last_linear = find_first_last_linear_layers(model)
    if module_filter_func is None:
        module_filter_func = partial(filter_linear_layers, first_layer_name=first_linear, last_layer_name=last_linear)
    convert_to_float8_training(model, module_filter_fn=module_filter_func, config=config)
