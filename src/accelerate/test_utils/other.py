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

import importlib
import os
from pathlib import Path
from typing import Callable, Dict

import torch


# Set a backend environment variable for any extra module import required for a custom hardware accelerator
if "ACCELERATE_TEST_BACKEND" in os.environ:
    backend = os.environ["ACCELERATE_TEST_BACKEND"]
    try:
        _ = importlib.import_module(backend)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to import `ACCELERATE_TEST_BACKEND` '{backend}'! This should be the name of an installed module \
                to enable a specified backend.):\n{e}"
        ) from e

if "ACCELERATE_TEST_DEVICE" in os.environ:
    torch_device = os.environ["ACCELERATE_TEST_DEVICE"]
    try:
        # try creating device to see if provided device is valid
        _ = torch.device(torch_device)
    except RuntimeError as e:
        raise RuntimeError(
            f"Unknown testing device specified by environment variable `ACCELERATE_TEST_DEVICE`: {torch_device}"
        ) from e
else:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"


# Utils for custom and alternative hardware accelerator devices

# Mappings from device names to callable functions to support device agnostic testing.
BACKEND_DEVICE_COUNT = {"cuda": torch.cuda.device_count, "cpu": lambda: 0, "default": lambda: 1}
BACKEND_IS_AVAILABLE = {"cuda": torch.cuda.is_available, "cpu": lambda: False, "default": torch.cuda.is_available}


# This dispatches a defined function according to the hardware accelerator from the function definitions.
def device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    return dispatch_table.get(device, dispatch_table["default"])(*args, **kwargs)


# These are callables which automatically dispatch the function specific to the hardware accelerator
def backend_device_count(device: str):
    return device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


def backend_is_available(device: str):
    return device_agnostic_dispatch(device, BACKEND_IS_AVAILABLE)


# Update device function dict mapping
def update_mapping_from_spec(dispatch_table: Dict[str, Callable], attribute_name: str):
    try:
        # Try to import the function directly
        spec_fn = getattr(device_spec_module, attribute_name)
        dispatch_table[torch_device] = spec_fn
    except AttributeError as e:
        # If the function doesn't exist, and there is no default, throw an error
        if "default" not in dispatch_table:
            raise AttributeError(
                f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
            ) from e


if "ACCELERATE_TEST_DEVICE_SPEC" in os.environ:
    device_spec_path = os.environ["ACCELERATE_TEST_DEVICE_SPEC"]
    if not Path(device_spec_path).is_file():
        raise ValueError(f"Specified path to device specification file is not found. Received {device_spec_path}")

    try:
        import_name = device_spec_path[: device_spec_path.index(".py")]
    except ValueError as e:
        raise ValueError(f"Provided device spec file is not a Python file! Received {device_spec_path}") from e

    device_spec_module = importlib.import_module(import_name)

    try:
        device_name = device_spec_module.DEVICE_NAME
    except AttributeError:
        raise AttributeError("Device spec file did not contain `DEVICE_NAME`")

    if "ACCELERATE_TEST_DEVICE" in os.environ and torch_device != device_name:
        msg = f"Mismatch between environment variable `ACCELERATE_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
        msg += "Either unset `ACCELERATE_TEST_DEVICE` or ensure it matches device spec name."
        raise ValueError(msg)

    torch_device = device_name

    # Add one entry here for each `BACKEND_*` dictionary.
    update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")
    update_mapping_from_spec(BACKEND_IS_AVAILABLE, "IS_AVAILABLE_FN")
