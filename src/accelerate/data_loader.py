import math
import random
import torch
from typing import Optional

from torch.utils.data import BatchSampler, DataLoader

from .config import DistributedState, DistributedType, is_tpu_available
from .utils import send_to_device, set_seed, synchronize_rng_states

if is_tpu_available():
    import torch_xla.distributed.parallel_loader as pl

class BatchSamplerShard(BatchSampler):
    """
    Wraps a PyTorch :obj:`BatchSampler` to generate batches for one of the processes only. Instances of this class will
    always yield a number of batches that is a round multiple of :obj:`num_processes` and that all have the same size.
    Depending on the value of the :obj:`drop_last` attribute of the batch sampler passed, it will either stop the
    iteration at the first batch that would be too small / not present on all processes or loop with indices from the
    beginning.
    
    Args:
        batch_sampler (:obj:`BatchSampler`):
            The batch sampler to split in several shards.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the shards should be created by splitting a batch to give a piece of it on each process, or by
            yielding different full batches on each process.
            
            On two processes with a sampler of :obj:`[[0, 1, 2, 3], [4, 5, 6, 7]]`, this will result in:
            
            - the sampler on process 0 to yield :obj:`[0, 1, 2, 3]` and the sampler on process 1 to yield :obj:`[4, 5,
              6, 7]` if this argument is set to :obj:`False`.
            - the sampler on process 0 to yield :obj:`[0, 1]` then :obj:`[4, 5]` and the sampler on process 1 to yield
              :obj:`[2, 3]` then :obj:`[6, 7]` if this argument is set to :obj:`True`.
    
    .. warning::
    
        This does not support :obj:`BatchSampler` with varying batch size yet.
    """
    
    def __init__(self, batch_sampler: BatchSampler, num_processes: int = 1, process_index: int = 0, split_batches: bool = False):
        if split_batches and batch_sampler.batch_size % num_processes != 0:
            raise ValueError(
                f"To use `BatchSamplerShard` in `split_batches` mode, the batch size ({batch_sampler.batch_size}) " 
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.batch_sampler = batch_sampler
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last
    
    def __len__(self):
        length = (len(self.batch_sampler) + self.num_processes - 1) // self.num_processes
        return length if self.drop_last else length + 1
    
    def __iter__(self):
        return self._iter_with_split() if self.split_batches else self._iter_with_no_split()
    
    def _iter_with_split(self):
        initial_data = []
        for idx, batch in enumerate(self.batch_sampler):
            if idx == 0:
                initial_data = batch
            if len(batch) == self.batch_size:
                # If the batch is full, we yield the part of it this process is responsible of.
                yield batch[self.num_processes * self.process_index: self.num_processes * (self.process_index + 1)]
        
        # If drop_last is True of the last batch was full, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0 and len(batch) < self.batch_size:
            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.batch_size:
                initial_data += initial_data
            batch = batch + initial_data
            yield batch[self.num_processes * self.process_index: self.num_processes * (self.process_index + 1)]
        
    def _iter_with_no_split(self):
        initial_data = []
        batch_to_yield = []
        for idx, batch in enumerate(self.batch_sampler):
            # We gather the initial indices in case we need to circle back at the end.
            if not self.drop_last and idx < self.num_processes:
                initial_data += batch
            # We identify the batch to yield but wait until we ar sure every process gets a full batch before actually
            # yielding it.
            if idx % self.num_processes == self.process_index:
                batch_to_yield = batch
            if idx % self.num_processes == self.num_processes - 1 and len(batch) == self.batch_size:
                yield batch_to_yield
                batch_to_yield = []
        
        # If drop_last is True, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0:
            # ... we yield the complete batch we had saved before if it has the proper length
            if len(batch_to_yield) == self.batch_size:
                yield batch_to_yield

            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.num_processes * self.batch_size:
                initial_data += initial_data
            
            # If the last batch seen was of the proper size, it has been yielded by its process so we move to the next
            if len(batch) == self.batch_size:
                batch = []
                idx += 1
            
            # Make sure we yield a multiple of self.num_processes batches
            cycle_index = 0
            while idx % self.num_processes != 0 or len(batch) > 0:
                end_index = cycle_index + self.batch_size - len(batch)
                batch += initial_data[cycle_index: end_index]
                if idx % self.num_processes == self.process_index:
                    yield batch
                cycle_index = end_index
                batch = []
                idx += 1


class DataLoaderShard(DataLoader):
    def __init__(self, dataset, device=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
    
    def __iter__(self):
        synchronize_rng_states()
        for batch in super().__iter__():
            yield batch if self.device is None else send_to_device(batch, self.device)

def prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: int = 1,
    process_index: int = 0,
    split_batches_across_devices: bool = False,
    put_on_device: bool = False,
) -> DataLoader:
    """
    Wraps a PyTorch :obj:`DataLoader` to generate batches for one of the processes only. 
    
    Depending on the value of the :obj:`drop_last` attribute of the :obj:`dataloader` passed, it will either stop the
    iteration at the first batch that would be too small / not present on all processes or loop with indices from the
    beginning.
    
    Args:
        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        device (:obj:`torch.device`):
            The target device for the returned :obj:`DataLoader`.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        split_batches_across_devices (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the resulting :obj:`DataLoader` should split the batches of the original data loader across devices
            or yield full batches (in which case it will yield batches starting at the :obj:`process_index`-th and
            advancing of :obj:`num_processes` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial :obj:`dataloader`
            if this option is set to :obj:`True`, the batch size of the initial :obj:`dataloader` multiplied by
            :obj:`num_processes` otherwise.

            Setting this option to :obj:`True` requires that the batch size of the :obj:`dataloader` is a round
            multiple of :obj:`batch_size`.
        put_on_device (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to put the batches on :obj:`device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
    
    Returns:
        :obj:`torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    .. warning::
    
        This does not support :obj:`BatchSampler` with varying batch size yet or :obj:`IterableDataset` yet.
    """
    if split_batches_across_devices and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"Using `split_batches_across_devices=True` requires that the batch size ({dataloader.batch_size}) " 
            f"to be a round multiple of the number of processes ({num_processes})."
        )
    if num_processes == 1:
        return dataloader
    new_batch_sampler = BatchSamplerShard(
        dataloader.batch_sampler,
        num_processes=num_processes,
        process_index=process_index,
        split_batches=split_batches_across_devices,
    )
    dataloader = DataLoaderShard(
        dataloader.dataset,
        device=device if put_on_device else None,
        batch_sampler=new_batch_sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        # prefetch_factor=dataloader.prefetch_factor,
        # persistent_workers=dataloader.persistent_workers,

    )
    if is_tpu_available():
        # TODO: check if this works:
        # return pl.MpDeviceLoader(dataloader, device)
        return pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
    return dataloader
