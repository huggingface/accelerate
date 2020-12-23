import random

import numpy as np
import torch

from .config import AcceleratorState, DistributedType, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


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


def synchronize_rng_states():
    """
    Helper function to synchronize the rng states in distributed / TPU training.
    """
    state = AcceleratorState()
    if state.distributed_type == DistributedType.TPU:
        rng_state = torch.get_rng_state()
        rng_state = xm.mesh_reduce("random_seed", rng_state, lambda x: x[0])
        torch.set_rng_state(rng_state)
    elif state.distributed_type == DistributedType.MULTI_GPU:
        rng_state = torch.get_rng_state().to(state.device)
        # Broadcast the state from process 0 to all the others.
        torch.distributed.broadcast(rng_state, 0)
        torch.set_rng_state(rng_state.cpu())

        # Broadcast the state from process 0 to all the others.
        rng_state = torch.cuda.get_rng_state().to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        torch.cuda.set_rng_state(rng_state.cpu())


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        raise TypeError(
            f"Can't send the values of type {type(tensor)} to device {device}, only of nested list/tuple/dicts "
            "of tensors or objects having a `to` method."
        )
    return tensor.to(device)


def extract_model_from_parallel(model):
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model
