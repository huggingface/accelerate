import torch

from .config import DistributedState, DistributedType, is_tpu_available

if is_tpu_available():
    import torch_xla.core.xla_model as xm


def _tpu_gather(tensor, name="tensor"):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_tpu_gather(t, name=f"{name}_{i}") for i, t in enumerate (tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _tpu_gather(v, name=f"{name}_{k}") for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    return xm.mesh_reduce(name, t, torch.cat)


def _gpu_gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_gpu_gather(t) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _gpu_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


def gather(tensor, name=None):
    """Gather tensor from all devices."""
    if DistributedState().distributed_type == DistributedType.TPU:
        return _tpu_gather(tensor, name="tensor" if name is None else name)
    elif DistributedState().distributed_type == DistributedType.MULTI_GPU:
        return _gpu_gather(tensor)
    else:
        return tensor