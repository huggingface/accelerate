# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch

from .state import AcceleratorState, DistributedType, is_deepspeed_available, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


def is_boto3_available():
    return importlib.util.find_spec("boto3") is not None


def is_sagemaker_available():
    return importlib.util.find_spec("sagemaker") is not None


if is_deepspeed_available():
    from deepspeed import DeepSpeedEngine


class RNGType(Enum):
    TORCH = "torch"
    CUDA = "cuda"
    XLA = "xla"
    GENERATOR = "generator"


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if is_tpu_available():
        xm.set_rng_state(seed)


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    # Get the proper rng state
    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = torch.cuda.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_tpu_available(), "Can't synchronize XLA seeds on an environment without TPUs."
        rng_state = torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        rng_state = generator.get_state()

    # Broadcast the rng state from device 0 to other devices
    state = AcceleratorState()
    if state.distributed_type == DistributedType.TPU:
        rng_state = xm.mesh_reduce("random_seed", rng_state, lambda x: x[0])
    elif state.distributed_type == DistributedType.MULTI_GPU:
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)

    # Set the broadcast rng state
    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        torch.cuda.set_rng_state(rng_state)
    elif rng_type == RNGType.XLA:
        xm.set_rng_state(rng_state.item())
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)


def synchronize_rng_states(rng_types: List[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    # There is no direct check whether an object if of type namedtuple sadly, this is a workaround.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # Can instantiate a namedtuple from a generator directly, contrary to a tuple/list.
        return type(obj)(*list(generator))
    return type(obj)(generator)


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (send_to_device(t, device) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def convert_to_fp32(tensor):
    """
    Recursively converts the lements nested list/tuple/dictionary of tensors in FP16 precision to FP32.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to convert from FP16 to FP32.

    Returns:
        The same data structure as :obj:`tensor` with all tensors that were in FP16 precision converted to FP32.
    """
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (convert_to_fp32(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: convert_to_fp32(v) for k, v in tensor.items()})
    elif not hasattr(tensor, "dtype") or tensor.dtype != torch.float16:
        return tensor
    return tensor.float()


def convert_outputs_to_fp32(model_forward):
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP16
    precision will be convert back to FP32.

    Args:
        model_forward (:obj:`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as :obj:`model_forward` but with converted outputs.
    """

    def convert_outputs(*args, **kwargs):
        outputs = model_forward(*args, **kwargs)
        return convert_to_fp32(outputs)

    return convert_outputs


def extract_model_from_parallel(model):
    """
    Extract a model from its distributed containers.

    Args:
        model (:obj:`torch.nn.Module`): The model to extract.

    Returns:
        :obj:`torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    if is_deepspeed_available():
        options += (DeepSpeedEngine,)

    while isinstance(model, options):
        model = model.module
    return model


def _tpu_gather(tensor, name="tensor"):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_tpu_gather(t, name=f"{name}_{i}") for i, t in enumerate(tensor)))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _tpu_gather(v, name=f"{name}_{k}") for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    return xm.mesh_reduce(name, tensor, torch.cat)


def _gpu_gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_gpu_gather(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _gpu_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


_cpu_gather = _gpu_gather


def gather(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor, name="accelerate.utils.gather")
    elif AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        return _gpu_gather(tensor)
    elif AcceleratorState().distributed_type == DistributedType.MULTI_CPU:
        return _cpu_gather(tensor)
    else:
        return tensor


def pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so they
    can safely be gathered.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to gather.
        dim (:obj:`int`, `optional`, defaults to 0):
            The dimension on which to pad.
        pad_index (:obj:`int`, `optional`, defaults to 0):
            The value with which to pad.
        pad_first (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to pad at the beginning or the end.
    """
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (pad_across_processes(t, dim=dim, pad_index=pad_index) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: pad_across_processes(v, dim=dim, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")

    if dim >= len(tensor.shape):
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = gather(size).cpu()
    # Then pad to the maximum size
    max_size = max(s[dim] for s in sizes)
    if max_size == tensor.shape[dim]:
        return tensor

    old_size = tensor.shape
    new_size = list(old_size)
    new_size[dim] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    if pad_first:
        indices = tuple(
            slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))
        )
    else:
        indices = tuple(slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size)))
    new_tensor[indices] = tensor
    return new_tensor


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    Warning::

        Make sure all processes will reach this instruction otherwise one of your processes will hang forever.
    """
    if (
        AcceleratorState().distributed_type == DistributedType.MULTI_GPU
        or AcceleratorState().distributed_type == DistributedType.MULTI_CPU
        or AcceleratorState().distributed_type == DistributedType.DEEPSPEED
    ):
        torch.distributed.barrier()
    elif AcceleratorState().distributed_type == DistributedType.TPU:
        xm.rendezvous("accelerate.utils.wait_for_everyone")


def save(obj, f):
    """
    Save the data to disk. Use in place of :obj:`torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
    """
    if AcceleratorState().distributed_type == DistributedType.TPU:
        xm.save(obj, f)
    elif AcceleratorState().local_process_index == 0:
        torch.save(obj, f)


class PrepareForLaunch:
    """
    Prepare a function that will launched in a distributed setup.

    Args:
        launcher (:obj:`Callable`):
            The function to launch.
        distributed_type (:class:`~accelerate.state.DistributedType`):
            The distributed type to prepare for.
    """

    def __init__(self, launcher, distributed_type="NO"):
        self.launcher = launcher
        self.distributed_type = DistributedType(distributed_type)

    def __call__(self, index, *args):
        if self.distributed_type == DistributedType.MULTI_GPU or self.distributed_type == DistributedType.MULTI_CPU:
            # Prepare the environment for torch.distributed
            os.environ["LOCAL_RANK"] = str(index)
            os.environ["RANK"] = str(index)

        self.launcher(*args)


@dataclass
class DeepSpeedPlugin:

    gradient_accumulation_steps: int = field(
        default=None, metadata={"help": "Number of steps to accumulate gradients before updating optimizer states"}
    )
    zero_stage: int = field(
        default=None,
        metadata={"help": "Possible options are 0,1,2,3; Default will be taken from environment variable"},
    )
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"},
    )

    auto_opt_mapping: bool = field(
        default=True,
        metadata={"help": "whether to map torch.adam to deepspeed optimizer version of adam based on config"},
    )

    offload_optimizer_device: bool = field(default=None, metadata={"help": "Possible options are none|cpu|nvme"})

    def __post_init__(self):

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))

        if self.zero_stage is None:
            self.zero_stage = int(os.environ.get("DEEPSPEED_ZERO_STAGE", 2))

        if self.offload_optimizer_device is None:
            self.offload_optimizer_device = os.environ.get("DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", "none")

        self.deepspeed_config = {
            "train_batch_size": None,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": self.offload_optimizer_device,
                },
            },
            "steps_per_print": float("inf"),  # this will stop deepspeed from logging @ stdout
        }
