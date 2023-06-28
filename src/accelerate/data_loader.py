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

import math
from contextlib import suppress
from typing import List, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_tpu_available
from .utils import (
    RNGType,
    broadcast,
    broadcast_object_list,
    concatenate,
    find_batch_size,
    get_data_structure,
    initialize_tensors,
    is_torch_version,
    send_to_device,
    slice_tensors,
    synchronize_rng_states,
)


logger = get_logger(__name__)

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
    "generator": None,
}

# kwargs added after by version
_PYTORCH_DATALOADER_ADDITIONAL_KWARGS = {
    "1.7.0": {"prefetch_factor": 2, "persistent_workers": False},
}

for v, additional_kwargs in _PYTORCH_DATALOADER_ADDITIONAL_KWARGS.items():
    if is_torch_version(">=", v):
        _PYTORCH_DATALOADER_KWARGS.update(additional_kwargs)


class BatchSamplerShard(BatchSampler):
    """
    Wraps a PyTorch `BatchSampler` to generate batches for one of the processes only. Instances of this class will
    always yield a number of batches that is a round multiple of `num_processes` and that all have the same size.
    Depending on the value of the `drop_last` attribute of the batch sampler passed, it will either stop the iteration
    at the first batch that would be too small / not present on all processes or loop with indices from the beginning.

    Args:
        batch_sampler (`torch.utils.data.sampler.BatchSampler`):
            The batch sampler to split in several shards.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the shards should be created by splitting a batch to give a piece of it on each process, or by
            yielding different full batches on each process.

            On two processes with a sampler of `[[0, 1, 2, 3], [4, 5, 6, 7]]`, this will result in:

            - the sampler on process 0 to yield `[0, 1, 2, 3]` and the sampler on process 1 to yield `[4, 5, 6, 7]` if
              this argument is set to `False`.
            - the sampler on process 0 to yield `[0, 1]` then `[4, 5]` and the sampler on process 1 to yield `[2, 3]`
              then `[6, 7]` if this argument is set to `True`.
        even_batches (`bool`, *optional*, defaults to `True`):
            Whether or not to loop back at the beginning of the sampler when the number of samples is not a round
            multiple of (original batch size / number of processes).

    <Tip warning={true}>

    `BatchSampler`s with varying batch sizes are not enabled by default. To enable this behaviour, set `even_batches`
    equal to `False`

    </Tip>"""

    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_processes: int = 1,
        process_index: int = 0,
        split_batches: bool = False,
        even_batches: bool = True,
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
        self.even_batches = even_batches
        self.batch_size = getattr(batch_sampler, "batch_size", None)
        self.drop_last = getattr(batch_sampler, "drop_last", False)
        if self.batch_size is None and self.even_batches:
            raise ValueError("You need to use `even_batches=False` when the batch sampler has no batch size.")

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        if self.split_batches:
            # Split batches does not change the length of the batch sampler
            return len(self.batch_sampler)
        if len(self.batch_sampler) % self.num_processes == 0:
            # If the length is a round multiple of the number of processes, it's easy.
            return len(self.batch_sampler) // self.num_processes
        length = len(self.batch_sampler) // self.num_processes
        if self.drop_last:
            # Same if we drop the remainder.
            return length
        elif self.even_batches:
            # When we even batches we always get +1
            return length + 1
        else:
            # Otherwise it depends on the process index.
            return length + 1 if self.process_index < len(self.batch_sampler) % self.num_processes else length

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
            if not self.even_batches:
                if len(batch) > batch_length * self.process_index:
                    yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]
            else:
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
            if idx % self.num_processes == self.num_processes - 1 and (
                self.batch_size is None or len(batch) == self.batch_size
            ):
                yield batch_to_yield
                batch_to_yield = []

        # If drop_last is True, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0:
            if not self.even_batches:
                if len(batch_to_yield) > 0:
                    yield batch_to_yield
            else:
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


if is_torch_version(">=", "1.11.0"):
    from torch.utils.data import IterDataPipe

    # Inherit from `IterDataPipe` so that we can benefit from `DataLoader`'s forward compatibilities
    # (e.g. use synced seeds to shuffle the datapipe every epoch)
    IterableDatasetShardBaseClass = IterDataPipe
else:
    IterableDatasetShardBaseClass = IterableDataset


class IterableDatasetShard(IterableDatasetShardBaseClass):
    """
    Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (depending of the value of
    `split_batches`, this is either `batch_size` or `batch_size x num_processes`). Depending on the value of the
    `drop_last` attribute of the batch sampler passed, it will either stop the iteration at the first batch that would
    be too small or loop with indices from the beginning.

    Args:
        dataset (`torch.utils.data.dataset.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard (if `split_batches=False`) or the size of the batches (if
            `split_batches=True`).
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the shards should be created by splitting a batch to give a piece of it on each process, or by
            yielding different full batches on each process.

            On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7]`, this will result in:

            - the shard on process 0 to yield `[0, 1, 2, 3]` and the shard on process 1 to yield `[4, 5, 6, 7]` if this
              argument is set to `False`.
            - the shard on process 0 to yield `[0, 1, 4, 5]` and the sampler on process 1 to yield `[2, 3, 6, 7]` if
              this argument is set to `True`.
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
        if split_batches and batch_size > 1 and batch_size % num_processes != 0:
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


class DataLoaderStateMixin:
    """
    Mixin class that adds a state to a `DataLoader` to keep track of the status inside the dataloader such as at the
    end of the iteration, the number of items in the dataset in the last batch relative to the batch size, and other
    useful information that might be needed.

    **Available attributes:**

        - **end_of_dataloader** (`bool`) -- Whether at the last iteration or batch
        - **remainder** (`int`) -- The number of items that are remaining in the last batch, relative to the total
          batch size

    """

    def __init_subclass__(cls, **kwargs):
        cls.end_of_dataloader = False
        cls.remainder = -1

    def reset(self):
        self.end_of_dataloader = False
        self.remainder = -1


class _DataLoaderWrapper:
    dataloader: DataLoader
    wrapper_attrs: tuple

    def __getattr__(self, name, *args, **kwargs):
        # NOTE `__getattr__` is only called when `name` is not found.
        try:
            return getattr(self.dataloader, name, *args, **kwargs)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute `{name}`")

    def __setattr__(self, attr, val):
        if attr in self.wrapper_attrs:
            self.__dict__[attr] = val
        setattr(self.dataloader, attr, val)

    def __len__(self):
        # Make `len()` work with dataloader wrapper
        return len(self.dataloader)


class DataLoaderShard(_DataLoaderWrapper, DataLoaderStateMixin):
    """
    Wrapper for a PyTorch `DataLoader` that will deal with device placement and current distributed setup.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The dataloader to be wrapped.
        device (`torch.device`, *optional*):
            If passed, the device to put all batches on.
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: an optional `torch.Generator`
        synchronized_generator (`torch.Generator`, *optional*):
            A random number generator to keep synchronized across processes.
        split_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            No longer in use

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    wrapper_attrs = (
        "end_of_dataloader",
        "remainder",
        "dataloader",
        "device",
        "rng_types",
        "synchronized_generator",
        "skip_batches",
        "gradient_state",
    )

    def __init__(self, dataloader, device=None, rng_types=None, synchronized_generator=None, skip_batches=0, **kwargs):
        self.dataloader = dataloader
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.reset()
        self.gradient_state._add_dataloader(self)
        # We can safely pass because the default is -1
        with suppress(Exception):
            length = getattr(self.dataset, "total_dataset_length", len(self.dataset))
            self.remainder = length % self.total_batch_size
        dataloader_iter = iter(self.dataloader)
        # We iterate one batch ahead to check when we are at the end
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            yield

        batch_index = 0
        while True:
            try:
                # But we still move it to the device so it is done before `StopIteration` is reached
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device)
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                if batch_index >= self.skip_batches:
                    yield current_batch
                break
        self.gradient_state._remove_dataloader(self)

    @property
    def total_batch_size(self):
        batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        return (
            batch_sampler.batch_size
            if getattr(batch_sampler, "split_batches", False)
            else (batch_sampler.batch_size * getattr(batch_sampler, "num_processes", 1))
        )

    @property
    def total_dataset_length(self):
        if hasattr(self.dataset, "total_length"):
            return self.dataset.total_length
        else:
            return len(self.dataset)


if is_tpu_available(check_device=False):
    import torch_xla.distributed.parallel_loader as xpl

    class MpDeviceLoaderWrapper(xpl.MpDeviceLoader):
        """
        Wrapper for the xpl.MpDeviceLoader class that knows the total batch size.

        XLA preloading threads will all call DataLoaderShard's __iter__(). Remove rng_types from DataLoaderShard to
        prevent it from using the XLA device in the preloading threads, and synchronize the RNG once from the main
        thread only.

        **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
        """

        def __init__(self, dataloader: DataLoaderShard, device: torch.device):
            super().__init__(dataloader, device)
            self._rng_types = self._loader.rng_types
            self._loader.rng_types = None

        def __iter__(self):
            if self._rng_types is not None:
                synchronize_rng_states(self._rng_types, self._loader.synchronized_generator)

            return super().__iter__()

        @property
        def total_batch_size(self):
            return self._loader.total_batch_size

        @property
        def total_dataset_length(self):
            return self._loader.total_dataset_length


class DataLoaderDispatcher(_DataLoaderWrapper, DataLoaderStateMixin):
    """
    Wrapper for a PyTorch `DataLoader` that will iterate and preprocess on process 0 only, then dispatch on each
    process their part of the batch.

    Args:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration). Another way to see this is that the observed batch size will be
            the same as the initial `dataloader` if this option is set to `True`, the batch size of the initial
            `dataloader` multiplied by `num_processes` otherwise. Setting this option to `True` requires that the batch
            size of the `dataloader` is a round multiple of `batch_size`.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning of an iteration.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    wrapper_attrs = (
        "end_of_dataloader",
        "remainder",
        "dataloader",
        "split_batches",
        "gradient_state",
        "state",
        "_drop_last",
        "skip_batches",
    )

    def __init__(self, dataloader, split_batches: bool = False, skip_batches=0, _drop_last: bool = False, **kwargs):
        if is_torch_version("<", "1.8.0"):
            raise ImportError(
                f"Using `DataLoaderDispatcher` requires PyTorch 1.8.0 minimum. You have {torch.__version__}."
            )
        self.dataloader = dataloader
        self.split_batches = split_batches
        self.gradient_state = GradientState()
        self.state = AcceleratorState()
        self._drop_last = _drop_last
        self.skip_batches = skip_batches

    def _fetch_batches(self, iterator):
        batches, batch = None, None
        # On process 0, we gather the batch to dispatch.
        if self.state.process_index == 0:
            try:
                if self.split_batches:
                    # One batch of the main iterator is dispatched and split.
                    batch = next(iterator)
                else:
                    # num_processes batches of the main iterator are concatenated then dispatched and split.
                    # We add the batches one by one so we have the remainder available when drop_last=False.
                    batches = []
                    for _ in range(self.state.num_processes):
                        batches.append(next(iterator))
                    batch = concatenate(batches, dim=0)
                # In both cases, we need to get the structure of the batch that we will broadcast on other
                # processes to initialize the tensors with the right shape.
                # data_structure, stop_iteration
                batch_info = [get_data_structure(batch), False]
            except StopIteration:
                batch_info = [None, True]
        else:
            batch_info = [None, self._stop_iteration]
        # This is inplace, so after this instruction, every process has the same `batch_info` as process 0.
        broadcast_object_list(batch_info)
        self._stop_iteration = batch_info[1]
        if self._stop_iteration:
            # If drop_last is False and split_batches is False, we may have a remainder to take care of.
            if not self.split_batches and not self._drop_last:
                if self.state.process_index == 0 and len(batches) > 0:
                    batch = concatenate(batches, dim=0)
                    batch_info = [get_data_structure(batch), False]
                else:
                    batch_info = [None, True]
                broadcast_object_list(batch_info)
        return batch, batch_info

    def __iter__(self):
        self.reset()
        self.gradient_state._add_dataloader(self)
        # We can safely pass because the default is -1
        with suppress(Exception):
            length = getattr(self.dataset, "total_dataset_length", len(self.dataset))
            self.remainder = length % self.total_batch_size
        # NOTE PyTorch DataLoader adds forward compatibilities for DataPipes, which broadcasts
        # shared seed to all dist processes. Thus, we need to create iterator for all dist processes.
        # But, we only iterate through the DataLoader on process 0.
        main_iterator = iter(self.dataloader)
        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        while not stop_iteration:
            batch, batch_info = next_batch, next_batch_info

            if self.state.process_index != 0:
                # Initialize tensors on other processes than process 0.
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(batch, self.state.device)
            # Broadcast the batch before splitting it.
            batch = broadcast(batch, from_process=0)

            if not self._drop_last and first_batch is None:
                # We keep at least num processes elements of the first batch to be able to complete the last batch
                first_batch = slice_tensors(batch, slice(0, self.state.num_processes))

            if batch is None:
                raise ValueError(
                    f"Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration."
                )

            observed_batch_size = find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes

            stop_iteration = self._stop_iteration
            if not stop_iteration:
                # We may still be at the end of the dataloader without knowing it yet: if there is nothing left in
                # the dataloader since the number of batches is a round multiple of the number of processes.
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                # next_batch_info[0] is None when there are no more batches, otherwise we still need to process them.
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True

            if not self._drop_last and stop_iteration and observed_batch_size % self.state.num_processes != 0:
                # If the last batch is not complete, let's add the first batch to it.
                batch = concatenate([batch, first_batch], dim=0)
                # Batch size computation above is wrong, it's off by 1 so we fix it.
                batch_size += 1

            data_slice = slice(self.state.process_index * batch_size, (self.state.process_index + 1) * batch_size)
            batch = slice_tensors(batch, data_slice)

            if stop_iteration:
                self.end_of_dataloader = True
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.gradient_state._remove_dataloader(self)

    def __len__(self):
        whole_length = len(self.dataloader)
        if self.split_batches:
            return whole_length
        elif self._drop_last:
            return whole_length // self.state.num_processes
        else:
            return math.ceil(whole_length / self.state.num_processes)

    @property
    def total_batch_size(self):
        return self.batch_size if self.split_batches else (self.batch_size * self.state.num_processes)

    @property
    def total_dataset_length(self):
        return len(self.dataset)


def prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[List[Union[str, RNGType]]] = None,
    dispatch_batches: Optional[bool] = None,
    even_batches: bool = True,
) -> DataLoader:
    """
    Wraps a PyTorch `DataLoader` to generate batches for one of the processes only.

    Depending on the value of the `drop_last` attribute of the `dataloader` passed, it will either stop the iteration
    at the first batch that would be too small / not present on all processes or loop with indices from the beginning.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        device (`torch.device`):
            The target device for the returned `DataLoader`.
        num_processes (`int`, *optional*):
            The number of processes running concurrently. Will default to the value given by
            [`~state.AcceleratorState`].
        process_index (`int`, *optional*):
            The index of the current process. Will default to the value given by [`~state.AcceleratorState`].
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial `dataloader` if
            this option is set to `True`, the batch size of the initial `dataloader` multiplied by `num_processes`
            otherwise.

            Setting this option to `True` requires that the batch size of the `dataloader` is a round multiple of
            `batch_size`.
        put_on_device (`bool`, *optional*, defaults to `False`):
            Whether or not to put the batches on `device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

        dispatch_batches (`bool`, *optional*):
            If set to `True`, the datalaoder prepared is only iterated through on the main process and then the batches
            are split and broadcast to each process. Will default to `True` when the underlying dataset is an
            `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.

    Returns:
        `torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    <Tip warning={true}>

    `BatchSampler`s with varying batch sizes are not enabled by default. To enable this behaviour, set `even_batches`
    equal to `False`

    </Tip>
    """
    if dispatch_batches is None:
        if is_torch_version("<", "1.8.0") or not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from AcceleratorState
    state = AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches and dataloader.batch_size > 1 and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
            f"needs to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    new_batch_sampler = dataloader.batch_sampler
    sampler_is_batch_sampler = False
    synchronized_generator = None
    # No change if no multiprocess
    if (num_processes != 1 or state.distributed_type == DistributedType.MEGATRON_LM) and not dispatch_batches:
        if isinstance(new_dataset, IterableDataset):
            synchronized_generator = getattr(dataloader.dataset, "generator", None)
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size if dataloader.batch_size else 1,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
            if new_batch_sampler is not None and split_batches:
                new_batch_sampler.batch_size = new_batch_sampler.batch_size // num_processes
        else:
            # New batch sampler for the current process.
            sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler) or dataloader.batch_sampler is None
            if sampler_is_batch_sampler:
                sampler = getattr(dataloader.sampler, "sampler", None)
            else:
                sampler = getattr(dataloader.batch_sampler, "sampler", None)
            if hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator()
                synchronized_generator = sampler.generator

            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            if batch_sampler is not None:
                new_batch_sampler = BatchSamplerShard(
                    batch_sampler,
                    num_processes=num_processes,
                    process_index=process_index,
                    split_batches=split_batches,
                    even_batches=even_batches,
                )

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    # NOTE Since we only wrap `IterableDataset` and the sampler/batch_sampler come with
    # `IterableDataset` are just placeholders, we can safely update `dataset`.
    _dataloader_forcibly_setattr(dataloader, "dataset", new_dataset)
    if new_batch_sampler is not None:
        if sampler_is_batch_sampler:
            _dataloader_forcibly_setattr(dataloader, "sampler", new_batch_sampler)
            if dataloader.batch_sampler is not None:
                # NOTE User cannot present both sampler and batch_sampler to `DataLoader`,
                # Thus, the `batch_sampler` in this case can only be the default `BatchSampler`
                # holding a reference to `sampler`
                dataloader.batch_sampler.sampler = new_batch_sampler
        else:
            _dataloader_forcibly_setattr(dataloader, "batch_sampler", new_batch_sampler)

    if dispatch_batches:
        dataloader = DataLoaderDispatcher(
            dataloader,
            split_batches=split_batches,
            _drop_last=dataloader.drop_last,
        )
    else:
        dataloader = DataLoaderShard(
            dataloader,
            device=device if put_on_device and state.distributed_type != DistributedType.TPU else None,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
        )

    if state.distributed_type == DistributedType.TPU:
        return MpDeviceLoaderWrapper(dataloader, device)
    return dataloader


def _dataloader_forcibly_setattr(dataloader, name, val):
    # NOTE After initialization, `DataLoader` prevents one from updating its internal attrs
    # We need to be careful of what we are doing here.
    super(DataLoader, dataloader).__setattr__(name, val)


class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`
    for the first epoch.
    """

    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples
        self.skip_batches = 0

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


class SkipDataLoader(_DataLoaderWrapper):
    """
    Wrapper for a PyTorch `DataLoader` that will skip the first batches for the first epoch.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The dataloader to be wrapped.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            No longer in use
    """

    wrapper_attrs = ("dataloader", "skip_batches")

    def __init__(self, dataloader, skip_batches=0, **kwargs):
        self.dataloader = dataloader
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, batch in enumerate(self.dataloader):
            if index >= self.skip_batches:
                yield batch
        self.skip_batches = 0


def skip_first_batches(dataloader, num_batches=0):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.
    """
    dataset = dataloader.dataset
    sampler_is_batch_sampler = False
    if isinstance(dataset, IterableDataset):
        new_batch_sampler = None
    else:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler) or dataloader.batch_sampler is None
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    if new_batch_sampler is not None:
        if sampler_is_batch_sampler:
            _dataloader_forcibly_setattr(dataloader, "sampler", new_batch_sampler)
            if dataloader.batch_sampler is not None:
                # NOTE User cannot present both sampler and batch_sampler to `DataLoader`,
                # Thus, the `batch_sampler` in this case can only be the default `BatchSampler`
                # holding a reference to `sampler`
                dataloader.batch_sampler.sampler = new_batch_sampler
        else:
            _dataloader_forcibly_setattr(dataloader, "batch_sampler", new_batch_sampler)

        return dataloader
    else:
        # Need to manually skip batches in the dataloader
        return SkipDataLoader(dataloader, skip_batches=num_batches)
