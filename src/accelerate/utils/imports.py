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
import importlib.metadata
import os
import warnings
from functools import lru_cache

import torch
from packaging import version
from packaging.version import parse

from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version


# Try to run Torch native job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = parse_flag_from_env("USE_TORCH_XLA", default=True)

_torch_xla_available = False
if USE_TORCH_XLA:
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        import torch_xla.runtime

        _torch_xla_available = True
    except ImportError:
        pass

# Keep it for is_tpu_available. It will be removed along with is_tpu_available.
_tpu_available = _torch_xla_available

# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()


def _is_package_available(pkg_name, metadata_name=None):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(pkg_name if metadata_name is None else metadata_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


def is_torch_distributed_available() -> bool:
    return _torch_distributed_available


def is_ccl_available():
    try:
        pass
    except ImportError:
        print(
            "Intel(R) oneCCL Bindings for PyTorch* is required to run DDP on Intel(R) GPUs, but it is not"
            " detected. If you see \"ValueError: Invalid backend: 'ccl'\" error, please install Intel(R) oneCCL"
            " Bindings for PyTorch*."
        )
    return (
        importlib.util.find_spec("torch_ccl") is not None
        or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
    )


def get_ccl_version():
    return importlib.metadata.version("oneccl_bind_pt")


def is_import_timer_available():
    return _is_package_available("import_timer")


def is_pynvml_available():
    return _is_package_available("pynvml") or _is_package_available("pynvml", "nvidia-ml-py")


def is_pytest_available():
    return _is_package_available("pytest")


def is_msamp_available():
    return _is_package_available("msamp", "ms-amp")


def is_schedulefree_available():
    return _is_package_available("schedulefree")


def is_transformer_engine_available():
    return _is_package_available("transformer_engine", "transformer-engine")


def is_lomo_available():
    return _is_package_available("lomo_optim")


def is_fp8_available():
    return is_msamp_available() or is_transformer_engine_available()


def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    pytorch_nvml_based_cuda_check_previous_value = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")
    try:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = str(1)
        available = torch.cuda.is_available()
    finally:
        if pytorch_nvml_based_cuda_check_previous_value:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = pytorch_nvml_based_cuda_check_previous_value
        else:
            os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)

    return available


@lru_cache
def is_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    warnings.warn(
        "`is_tpu_available` is deprecated and will be removed in v0.27.0. "
        "Please use the `is_torch_xla_available` instead.",
        FutureWarning,
    )
    # Due to bugs on the amp series GPUs, we disable torch-xla on them
    if is_cuda_available():
        return False
    if check_device:
        if _tpu_available:
            try:
                # Will raise a RuntimeError if no XLA configuration is found
                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
    return _tpu_available


@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    if not _torch_xla_available:
        return False
    elif check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


def is_deepspeed_available():
    if is_mlu_available():
        return _is_package_available("deepspeed", metadata_name="deepspeed-mlu")
    return _is_package_available("deepspeed")


def is_pippy_available():
    return is_torch_version(">=", "2.4.0")


def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    if is_torch_xla_available(check_is_tpu=True):
        return not ignore_tpu
    if is_cuda_available():
        return torch.cuda.is_bf16_supported()
    if is_mps_available():
        return False
    return True


def is_4bit_bnb_available():
    package_exists = _is_package_available("bitsandbytes")
    if package_exists:
        bnb_version = version.parse(importlib.metadata.version("bitsandbytes"))
        return compare_versions(bnb_version, ">=", "0.39.0")
    return False


def is_8bit_bnb_available():
    package_exists = _is_package_available("bitsandbytes")
    if package_exists:
        bnb_version = version.parse(importlib.metadata.version("bitsandbytes"))
        return compare_versions(bnb_version, ">=", "0.37.2")
    return False


def is_bnb_available():
    return _is_package_available("bitsandbytes")


def is_torchvision_available():
    return _is_package_available("torchvision")


def is_megatron_lm_available():
    if str_to_bool(os.environ.get("ACCELERATE_USE_MEGATRON_LM", "False")) == 1:
        if importlib.util.find_spec("megatron") is not None:
            try:
                megatron_version = parse(importlib.metadata.version("megatron-core"))
                if compare_versions(megatron_version, "==", "0.5.0"):
                    return importlib.util.find_spec(".data", "megatron")
            except Exception as e:
                warnings.warn(f"Parse Megatron version failed. Exception:{e}")
                return False


def is_transformers_available():
    return _is_package_available("transformers")


def is_datasets_available():
    return _is_package_available("datasets")


def is_peft_available():
    return _is_package_available("peft")


def is_timm_available():
    return _is_package_available("timm")


def is_triton_available():
    return _is_package_available("triton")


def is_aim_available():
    package_exists = _is_package_available("aim")
    if package_exists:
        aim_version = version.parse(importlib.metadata.version("aim"))
        return compare_versions(aim_version, "<", "4.0.0")
    return False


def is_tensorboard_available():
    return _is_package_available("tensorboard") or _is_package_available("tensorboardX")


def is_wandb_available():
    return _is_package_available("wandb")


def is_comet_ml_available():
    return _is_package_available("comet_ml")


def is_boto3_available():
    return _is_package_available("boto3")


def is_rich_available():
    if _is_package_available("rich"):
        if "ACCELERATE_DISABLE_RICH" in os.environ:
            warnings.warn(
                "`ACCELERATE_DISABLE_RICH` is deprecated and will be removed in v0.22.0 and deactivated by default. Please use `ACCELERATE_ENABLE_RICH` if you wish to use `rich`."
            )
            return not parse_flag_from_env("ACCELERATE_DISABLE_RICH", False)
        return parse_flag_from_env("ACCELERATE_ENABLE_RICH", False)
    return False


def is_sagemaker_available():
    return _is_package_available("sagemaker")


def is_tqdm_available():
    return _is_package_available("tqdm")


def is_clearml_available():
    return _is_package_available("clearml")


def is_pandas_available():
    return _is_package_available("pandas")


def is_mlflow_available():
    if _is_package_available("mlflow"):
        return True

    if importlib.util.find_spec("mlflow") is not None:
        try:
            _ = importlib.metadata.metadata("mlflow-skinny")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
    return False


def is_mps_available(min_version="1.12"):
    # With torch 1.12, you can use torch.backends.mps
    # With torch 2.0.0, you can use torch.mps
    return is_torch_version(">=", min_version) and torch.backends.mps.is_available() and torch.backends.mps.is_built()


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


@lru_cache
def is_mlu_available(check_device=False):
    "Checks if `torch_mlu` is installed and potentially if a MLU is in the environment"
    if importlib.util.find_spec("torch_mlu") is None:
        return False

    import torch_mlu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no MLU is found
            _ = torch.mlu.device_count()
            return torch.mlu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "mlu") and torch.mlu.is_available()


@lru_cache
def is_musa_available(check_device=False):
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if importlib.util.find_spec("torch_musa") is None:
        return False

    import torch_musa  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            _ = torch.musa.device_count()
            return torch.musa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available either via `intel_extension_for_pytorch` or via stock PyTorch (>=2.4) and
    potentially if a XPU is in the environment
    """

    "check if user disables it explicitly"
    if not parse_flag_from_env("ACCELERATE_USE_XPU", default=True):
        return False

    if is_ipex_available():
        if is_torch_version("<=", "1.12"):
            return False

        import intel_extension_for_pytorch  # noqa: F401
    else:
        if is_torch_version("<=", "2.3"):
            return False

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU  is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_dvclive_available():
    return _is_package_available("dvclive")


def is_torchdata_available():
    return _is_package_available("torchdata")


# TODO: Remove this function once stateful_dataloader is a stable feature in torchdata.
def is_torchdata_stateful_dataloader_available():
    package_exists = _is_package_available("torchdata")
    if package_exists:
        torchdata_version = version.parse(importlib.metadata.version("torchdata"))
        return compare_versions(torchdata_version, ">=", "0.8.0")
    return False
