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

from typing import List, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from packaging import version

from .state import AcceleratorState, DistributedType, is_tpu_available
from .utils import RNGType, send_to_device, synchronize_rng_states


if is_tpu_available():
    import torch_xla.core.xla_model as xm


# kwargs of the DataLoader in min version 1.4.0.
_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
}

# kwargs added after by version
_PYTORCH_DATALOADER_ADDITIONAL_KWARGS = {
    "1.6.0": {"generator": None},
    "1.7.0": {"prefetch_factor": 2, "persistent_workers": False},
}

for v, additional_kwargs in _PYTORCH_DATALOADER_ADDITIONAL_KWARGS.items():
    if version.parse(torch.__version__) >= version.parse(v):
        _PYTORCH_DATALOADER_KWARGS.update(additional_kwargs)


class BatchSamplerShard(BatchSampler):
    """
    Wraps a PyTorch :obj:`BatchSampler` to generate batches for one of the processes only. Instances of this class will
    always yield a number of batches that is a round multiple of :obj:`num_processes` and that all have the same size.
    Depending on the value of the :obj:`drop_last` attribute of the batch sampler passed, it will either stop the
    iteration at the first batch that would be too small / not present on all processes or loop with indices from the
    beginning.

    Args:
        batch_sampler (:obj:`torch.utils.data.sampler.BatchSampler`):
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

    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
    ):
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
        if self.split_batches:
            return len(self.batch_sampler)
        if len(self.batch_sampler) % self.num_processes == 0:
            return len(self.batch_sampler) // self.num_processes
        length = len(self.batch_sampler) // self.num_processes
        return length if self.drop_last else length + 1

    def __iter__(self):
        return self._iter_with_split() if self.split_batches else self._iter_with_no_split()

    def _iter_with_split(self):
        initial_data = []
        batch_length = self.batch_sampler.batch_size // self.num_processes
        for idx, batch in enumerate(self.batch_sampler):
            if idx == 0:
                initial_data = batch
            if len(batch) == self.batch_size:
                # If the batch is full, we yield the part of it this process is responsible of.
                yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]

        # If drop_last is True of the last batch was full, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0 and len(batch) < self.batch_size:
            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.batch_size:
                initial_data += initial_data
            batch = batch + initial_data
            yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]

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
                batch += initial_data[cycle_index:end_index]
                if idx % self.num_processes == self.process_index:
                    yield batch
                cycle_index = end_index
                batch = []
                idx += 1


class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch :obj:`IterableDataset` to generate samples for one of the processes only. Instances of this class
    will always yield a number of samples that is a round multiple of the actual batch size (depending of the value of
    :obj:`split_batches`, this is either :obj:`batch_size` or :obj:`batch_size x num_processes`). Depending on the
    value of the :obj:`drop_last` attribute of the batch sampler passed, it will either stop the iteration at the first
    batch that would be too small or loop with indices from the beginning.

    Args:
        dataset (:obj:`torch.utils.data.dataset.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The size of the batches per shard (if :obj:`split_batches=False`) or the size of the batches (if
            :obj:`split_batches=True`).
        drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the shards should be created by splitting a batch to give a piece of it on each process, or by
            yielding different full batches on each process.

            On two processes with an iterable dataset yielding of :obj:`[0, 1, 2, 3, 4, 5, 6, 7]`, this will result in:

            - the shard on process 0 to yield :obj:`[0, 1, 2, 3]` and the shard on process 1 to yield :obj:`[4, 5, 6,
              7]` if this argument is set to :obj:`False`.
            - the shard on process 0 to yield :obj:`[0, 1, 4, 5]` and the sampler on process 1 to yield :obj:`[2, 3, 6,
              7]` if this argument is set to :obj:`True`.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
    ):
        if split_batches and batch_size % num_processes != 0:
            raise ValueError(
                f"To use `IterableDatasetShard` in `split_batches` mode, the batch size ({batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches

    def __iter__(self):
        real_batch_size = self.batch_size if self.split_batches else (self.batch_size * self.num_processes)
        process_batch_size = (self.batch_size // self.num_processes) if self.split_batches else self.batch_size
        process_slice = range(self.process_index * process_batch_size, (self.process_index + 1) * process_batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]


class DataLoaderShard(DataLoader):
    """
    Subclass of a PyTorch :obj:`DataLoader` that will deal with device placement and current distributed setup.

    Args:
        dataset (:obj:`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        device (:obj:`torch.device`, `optional`):
            If passed, the device to put all batches on.
        rng_types (list of :obj:`str` or :class:`~accelerate.utils.RNGType`):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - :obj:`"torch"`: the base torch random number generator
            - :obj:`"cuda"`: the CUDA random number generator (GPU only)
            - :obj:`"xla"`: the XLA random number generator (TPU only)
            - :obj:`"generator"`: an optional :obj:`torch.Generator`
        generator (:obj:`torch.Generator`, `optional`):
            A random number generator to keep synchronized across processes.
        kwargs:
            All other keyword arguments to pass to the regular :obj:`DataLoader` initialization.
    """

    def __init__(self, dataset, device=None, rng_types=None, generator=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
        self.rng_types = rng_types
        self.generator = generator

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.generator)
        state = AcceleratorState()
        for batch in super().__iter__():
            if state.distributed_type == DistributedType.TPU:
                xm.mark_step()
            yield batch if self.device is None else send_to_device(batch, self.device)


def prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[List[Union[str, RNGType]]] = None,
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
        num_processes (:obj:`int`, `optional`):
            The number of processes running concurrently. Will default to the value given by
            :class:`~accelerate.AcceleratorState`.
        process_index (:obj:`int`, `optional`):
            The index of the current process. Will default to the value given by :class:`~accelerate.AcceleratorState`.
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
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
        rng_types (list of :obj:`str` or :class:`~accelerate.utils.RNGType`):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - :obj:`"torch"`: the base torch random number generator
            - :obj:`"cuda"`: the CUDA random number generator (GPU only)
            - :obj:`"xla"`: the XLA random number generator (TPU only)
            - :obj:`"generator"`: the :obj:`torch.Generator` of the sampler (or batch sampler if there is no sampler in
              your dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

    Returns:
        :obj:`torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    .. warning::

        This does not support :obj:`BatchSampler` with varying batch size yet.
    """
    # Grab defaults from AcceleratorState
    state = AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"Using `split_batches=True` requires that the batch size ({dataloader.batch_size}) "
            f"to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    generator = getattr(dataloader, "generator", None)
    # No change if no multiprocess
    if num_processes != 1:
        if isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            # New batch sampler for the current process.
            if hasattr(dataloader.sampler, "generator"):
                if dataloader.sampler.generator is None:
                    dataloader.sampler.generator = torch.Generator()
                    generator = dataloader.sampler.generator
                    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            elif getattr(dataloader.batch_sampler, "generator", None) is not None:
                generator = dataloader.batch_sampler.generator
            new_batch_sampler = BatchSamplerShard(
                dataloader.batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
        "generator",
    ]

    if rng_types is not None and generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["batch_size"] = dataloader.batch_size // num_processes if split_batches else dataloader.batch_size

    return DataLoaderShard(
        new_dataset,
        device=device if put_on_device else None,
        batch_sampler=new_batch_sampler,
        rng_types=rng_types,
        generator=generator,
        **kwargs,
    )
