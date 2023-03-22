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

import torch

from ..commands.config.default import write_basic_config  # noqa: F401
from ..state import PartialState
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_tpu_available
from .transformer_engine import convert_model


if is_deepspeed_available():
    from deepspeed import DeepSpeedEngine

if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    if is_deepspeed_available():
        options += (DeepSpeedEngine,)

    while isinstance(model, options):
        model = model.module

    if not keep_fp32_wrapper:
        forward = getattr(model, "forward")
        original_forward = model.__dict__.pop("_original_forward", None)
        if original_forward is not None:
            while hasattr(forward, "__wrapped__"):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = forward
        if getattr(model, "_converted_to_transformer_engine", False):
            convert_model(model, to_transformer_engine=False)
    return model


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    <Tip warning={true}>

    Make sure all processes will reach this instruction otherwise one of your processes will hang forever.

    </Tip>
    """
    PartialState().wait_for_everyone()


def save(obj, f):
    """
    Save the data to disk. Use in place of `torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
    """
    if PartialState().distributed_type == DistributedType.TPU:
        xm.save(obj, f)
    elif PartialState().local_process_index == 0:
        torch.save(obj, f)


@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.

    Will convert the values in `kwargs` to strings and upper-case all the keys.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import patch_environment

    >>> with patch_environment(FOO="bar"):
    ...     print(os.environ["FOO"])  # prints "bar"
    >>> print(os.environ["FOO"])  # raises KeyError
    ```
    """
    for key, value in kwargs.items():
        os.environ[key.upper()] = str(value)

    yield

    for key in kwargs:
        if key.upper() in os.environ:
            del os.environ[key.upper()]


def get_pretty_name(obj):
    """
    Gets a pretty name from `obj`.
    """
    if not hasattr(obj, "__qualname__") and not hasattr(obj, "__name__"):
        obj = getattr(obj, "__class__", obj)
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)


def merge_dicts(source, destination):
    """
    Recursively merges two dictionaries.

    Args:
        source (`dict`): The dictionary to merge into `destination`.
        destination (`dict`): The dictionary to merge `source` into.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dicts(value, node)
        else:
            destination[key] = value

    return destination
