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
from pathlib import Path

import torch

from ..commands.config.cluster import ClusterConfig
from ..commands.config.config_args import default_json_config_file
from ..state import AcceleratorState
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_tpu_available


if is_deepspeed_available():
    from deepspeed import DeepSpeedEngine

if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def extract_model_from_parallel(model):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`): The model to extract.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    if is_deepspeed_available():
        options += (DeepSpeedEngine,)

    while isinstance(model, options):
        model = model.module
    return model


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    <Tip warning={true}>

    Make sure all processes will reach this instruction otherwise one of your processes will hang forever.

    </Tip>
    """
    if (
        AcceleratorState().distributed_type == DistributedType.MULTI_GPU
        or AcceleratorState().distributed_type == DistributedType.MULTI_CPU
        or AcceleratorState().distributed_type == DistributedType.DEEPSPEED
        or AcceleratorState().distributed_type == DistributedType.FSDP
    ):
        torch.distributed.barrier()
    elif AcceleratorState().distributed_type == DistributedType.TPU:
        xm.rendezvous("accelerate.utils.wait_for_everyone")


def save(obj, f):
    """
    Save the data to disk. Use in place of `torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        xm.save(obj, f)
    elif AcceleratorState().local_process_index == 0:
        torch.save(obj, f)


@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.

    Will convert the values in `kwargs` to strings and upper-case all the keys.
    """
    for key, value in kwargs.items():
        os.environ[key.upper()] = str(value)

    yield

    for key in kwargs:
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


def write_basic_config(mixed_precision="no", save_location: str = default_json_config_file):
    """
    Creates and saves a basic cluster config to be used on a local machine with potentially multiple GPUs. Will also
    set CPU if it is a CPU-only machine.

    Args:
        mixed_precision (`str`, *optional*, defaults to "no"):
            Mixed Precision to use. Should be one of "no", "fp16", or "bf16"
        save_location (`str`, *optional*, defaults to `default_json_config_file`):
            Optional custom save location. Should be passed to `--config_file` when using `accelerate launch`. Default
            location is inside the huggingface cache folder (`~/.cache/huggingface`) but can be overriden by setting
            the `HF_HOME` environmental variable, followed by `accelerate/default_config.yaml`.
    """
    path = Path(save_location)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(
            f"Configuration already exists at {save_location}, will not override. Run `accelerate config` manually or pass a different `save_location`."
        )
        return
    mixed_precision = mixed_precision.lower()
    if mixed_precision not in ["no", "fp16", "bf16"]:
        raise ValueError(f"`mixed_precision` should be one of 'no', 'fp16', or 'bf16'. Received {mixed_precision}")
    config = {"compute_environment": "LOCAL_MACHINE", "mixed_precision": mixed_precision}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        config["num_processes"] = num_gpus
        config["use_cpu"] = False
        if num_gpus > 1:
            config["distributed_type"] = "MULTI_GPU"
        else:
            config["distributed_type"] = "NO"
    else:
        num_gpus = 0
        config["use_cpu"] = True
        config["num_processes"] = 1
        config["distributed_type"] = "NO"
    if not path.exists():
        config = ClusterConfig(**config)
        config.to_json_file(path)
