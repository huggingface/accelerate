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
from typing import TYPE_CHECKING, Callable, Optional

import torch

from .imports import is_torchao_available, torchao_required


if TYPE_CHECKING:
    if is_torchao_available():
        from torchao.float8.float8_linear import Float8LinearConfig


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


def filter_linear_layers(module, fqn: str, layers_to_filter: list[str]) -> bool:
    """
    A function which will check if `module` is:
    - a `torch.nn.Linear` layer
    - has in_features and out_features divisible by 16
    - is not part of `layers_to_filter`

    Args:
        module (`torch.nn.Module`):
            The module to check.
        fqn (`str`):
            The fully qualified name of the layer.
        layers_to_filter (`List[str]`):
            The list of layers to filter.
    """
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    if fqn in layers_to_filter:
        return False
    return True


def filter_first_and_last_linear_layers(module, fqn: str) -> bool:
    """
    A filter function which will filter out all linear layers except the first and last.

    <Tip>

        For stability reasons, we skip the first and last linear layers Otherwise can lead to the model not training or
        converging properly

    </Tip>

    Args:
        module (`torch.nn.Module`):
            The module to check.
        fqn (`str`):
            The fully qualified name of the layer.
    """
    first_linear, last_linear = find_first_last_linear_layers(module)
    return filter_linear_layers(module, fqn, layers_to_filter=[first_linear, last_linear])


@torchao_required
def has_ao_layers(model: torch.nn.Module):
    from torchao.float8.float8_linear import Float8Linear

    for name, module in model.named_modules():
        if isinstance(module, Float8Linear):
            return True
    return False


@torchao_required
def convert_model_to_fp8_ao(
    model: torch.nn.Module,
    config: Optional["Float8LinearConfig"] = None,
    module_filter_func: Optional[Callable] = filter_first_and_last_linear_layers,
):
    """
    Converts all `nn.Linear` layers in the model (except the first and last) to torchao's `Float8Linear` layer inplace.

    Args:
        model (`torch.nn.Module`):
            The model to convert.
        config (`torchao.float8.Float8LinearConfig`, *optional*):
            The configuration for the FP8 training. Recommended to utilize
            `torchao.float8.recipe_name_to_linear_config` to generate this. In general, the default config should be
            sufficient (what is passed when set to `None`).
        module_filter_func (`Callable`, *optional*, defaults to `filter_linear_layers`):
            Optional function that must take in a module and layer name, and returns a boolean indicating whether the
            module should be converted to FP8. Defaults to `filter_linear_layers`. See it for an example.

    Example:

    ```python
    from accelerate.utils.ao import convert_model_to_fp8_ao

    model = MyModel()
    model.to("cuda")
    convert_to_float8_training(model)

    model.train()
    ```
    """
    from torchao.float8 import convert_to_float8_training

    first_linear, last_linear = find_first_last_linear_layers(model)
    if module_filter_func is None:
        module_filter_func = partial(filter_linear_layers, layers_to_filter=[first_linear, last_linear])
    convert_to_float8_training(model, module_filter_fn=module_filter_func, config=config)
