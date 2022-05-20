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
import sys


# The package importlib_metadata is in a different place, depending on the Python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


try:
    import torch_ccl  # noqa: F401

    _ccl_available = True
except ImportError:
    _ccl_available = False


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    _tpu_available = True
except ImportError:
    _tpu_available = False


def is_ccl_available():
    return _ccl_available


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_tpu_available():
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


def is_tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None


def is_comet_ml_available():
    return importlib.util.find_spec("comet_ml") is not None


def is_boto3_available():
    return importlib.util.find_spec("boto3") is not None


def is_sagemaker_available():
    return importlib.util.find_spec("sagemaker") is not None
