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

import importlib
import os
import sys
import warnings
from distutils.util import strtobool
from functools import lru_cache

import torch

from packaging.version import parse

from .environment import parse_flag_from_env
from .versions import compare_versions, is_torch_version


# The package importlib_metadata is in a different place, depending on the Python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    _tpu_available = True
except ImportError:
    _tpu_available = False


def is_ccl_available():
    return (
        importlib.util.find_spec("torch_ccl") is not None
        or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
    )


def get_ccl_version():
    return importlib_metadata.version("oneccl_bind_pt")


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


@lru_cache()
def is_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    if _tpu_available and check_device:
        try:
            # Will raise a RuntimeError if no XLA configuration is found
            _ = xm.xla_device()
            return True
        except RuntimeError:
            return False
    return _tpu_available


def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None
    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    if is_tpu_available():
        return not ignore_tpu
    if is_torch_version(">=", "1.10"):
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
        return True
    return False


def is_megatron_lm_available():
    if strtobool(os.environ.get("ACCELERATE_USE_MEGATRON_LM", "False")) == 1:
        package_exists = importlib.util.find_spec("megatron") is not None
        if package_exists:
            megatron_version = parse(importlib_metadata.version("megatron-lm"))
            return compare_versions(megatron_version, ">=", "2.2.0")
    return False


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


def is_transformers_available():
    return importlib.util.find_spec("transformers") is not None


def is_datasets_available():
    return importlib.util.find_spec("datasets") is not None


def is_aim_available():
    return importlib.util.find_spec("aim") is not None


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None


def is_comet_ml_available():
    return importlib.util.find_spec("comet_ml") is not None


def is_boto3_available():
    return importlib.util.find_spec("boto3") is not None


def is_rich_available():
    if importlib.util.find_spec("rich") is not None:
        if parse_flag_from_env("DISABLE_RICH"):
            warnings.warn(
                "The `DISABLE_RICH` flag is deprecated and will be removed in version 0.17.0 of 🤗 Accelerate. Use `ACCELERATE_DISABLE_RICH` instead.",
                FutureWarning,
            )
            return not parse_flag_from_env("DISABLE_RICH")
        return not parse_flag_from_env("ACCELERATE_DISABLE_RICH")
    return False


def is_sagemaker_available():
    return importlib.util.find_spec("sagemaker") is not None


def is_tqdm_available():
    return importlib.util.find_spec("tqdm") is not None


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None
