import torch

from .config import DistributedState, DistributedType
from .data_loader import prepare_data_loader
from .optimizer import AcceleratedOptimizer


class Accelerator:
    def __init__(
        self,
        put_objects_on_device: bool = False,
        split_batches_across_devices: bool = False,
    ):
        self.distributed_state = DistributedState()
        self.put_objects_on_device = put_objects_on_device
        self.split_batches_across_devices = split_batches_across_devices

    @property
    def distributed_type(self):
        return self.distributed_state.distributed_type

    @property
    def num_processes(self):
        return self.distributed_state.num_processes

    @property
    def process_index(self):
        return self.distributed_state.process_index

    @property
    def local_rank(self):
        return self.distributed_state.local_rank

    @property
    def device(self):
        return self.distributed_state.device

    def is_main_process(self):
        return self.process_index == 0

    def print(self, *args, **kwargs):
        if self.is_main_process():
            print(*args, **kwargs)

    def _prepare_one(self, obj):
        if isinstance(obj, torch.utils.data.DataLoader):
            return self.prepare_data_loader(obj)
        elif isinstance(obj, torch.nn.Module):
            return self.prepare_model(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            return self.prepare_optimizer(obj)
        else:
            return obj

    def prepare(self, *args):
        return tuple(self._prepare_one(obj) for obj in args)

    def prepare_model(self, model):
        if self.put_objects_on_device:
            model = model.to(self.device)
        if self.distributed_type == DistributedType.MULTI_GPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        return model

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches_across_devices=self.split_batches_across_devices,
            put_on_device=self.put_objects_on_device,
        )

    def prepare_optimizer(self, optimizer):
        return AcceleratedOptimizer(optimizer)
