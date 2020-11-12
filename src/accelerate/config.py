import os
from enum import Enum

import torch


try:
    import torch_xla.core.xla_model as xm

    _tpu_available = True
except ImportError:
    _tpu_available = False


def is_tpu_available():
    return _tpu_available


class DistributedType(Enum):
    NO = 0
    MULTI_GPU = 1
    TPU = 2


# Inspired by Alex Martelli's 'Borg'.
class DistributedState:
    """
    This is a variation of a `singleton class <https://en.wikipedia.org/wiki/Singleton_pattern>`__ in the sense that
    all instance of :obj:`DistributedState` share the same state, which is initialized on the first instantiation.
    """

    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not getattr(self, "initialized", False):
            if is_tpu_available():
                self.distributed_type = DistributedType.TPU
                self.num_processes = xm.xrt_world_size()
                self.process_index = self.local_rank = xm.get_ordinal()
                self.device = xm.xla_device()
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1:
                self.distributed_type = DistributedType.MULTI_GPU
                torch.distributed.init_process_group(backend="nccl")
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_rank = 0
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.initialized = True

    def __repr__(self):
        return (
            f"Distributed environment: {self.distributed_type}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local rank: {self.local_rank}\n"
            f"Device: {self.device}"
        )
