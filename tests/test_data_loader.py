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

import random
import weakref

import pytest
import torch
from parameterized import parameterized
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from accelerate import Accelerator, PartialState
from accelerate.data_loader import (
    BatchSamplerShard,
    DataLoaderDispatcher,
    DataLoaderShard,
    DataLoaderStateMixin,
    IterableDatasetShard,
    SkipBatchSampler,
    SkipDataLoader,
    prepare_data_loader,
    skip_first_batches,
)
from accelerate.state import GradientState
from accelerate.test_utils.testing import AccelerateTestCase, require_torchdata_stateful_dataloader
from accelerate.utils import is_torchdata_stateful_dataloader_available, set_seed


if is_torchdata_stateful_dataloader_available():
    from torchdata.stateful_dataloader import (
        StatefulDataLoader,
    )


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = f"num_workers_{param.args[0]}"
    return f"{func.__name__}_{param_based_name}"


class RandomIterableDataset(IterableDataset):
    # For testing, an iterable dataset of random length
    def __init__(self, p_stop=0.01, max_length=1000):
        self.p_stop = p_stop
        self.max_length = max_length

    def __iter__(self):
        count = 0
        stop = False
        while not stop and count < self.max_length:
            yield count
            count += 1
            stop = random.random() < self.p_stop


class SimpleIterableDataset(IterableDataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield torch.rand(1)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SimpleBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, generator, seed):
        super().__init__(sampler, batch_size, drop_last)
        self.generator = generator
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()

    def set_epoch(self, epoch):
        self.epoch = epoch


class DataLoaderTester(AccelerateTestCase):
    def check_batch_sampler_shards(self, batch_sampler, expected, split_batches=False, even_batches=True):
        batch_sampler_shards = [
            BatchSamplerShard(batch_sampler, 2, i, split_batches=split_batches, even_batches=even_batches)
            for i in range(2)
        ]
        batch_sampler_lists = [list(batch_sampler_shard) for batch_sampler_shard in batch_sampler_shards]
        if not split_batches:
            assert [len(shard) for shard in batch_sampler_shards] == [len(e) for e in expected]
        assert batch_sampler_lists == expected

    def test_batch_sampler_shards_with_no_splits(self):
        # Check the shards when the dataset is a round multiple of total batch size.
        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is a round multiple of batch size but not total batch size.
        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [0, 1, 2]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is not a round multiple of batch size but has a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 0, 1]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is not a round multiple of batch size but and has not a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 0]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [1, 2, 3]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is very small.
        batch_sampler = BatchSampler(range(2), batch_size=3, drop_last=False)
        expected = [[[0, 1, 0]], [[1, 0, 1]]]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(2), batch_size=3, drop_last=True)
        expected = [[], []]
        self.check_batch_sampler_shards(batch_sampler, expected)

    def test_batch_sampler_shards_with_splits(self):
        # Check the shards when the dataset is a round multiple of batch size.
        batch_sampler = BatchSampler(range(24), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 21]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [22, 23]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        batch_sampler = BatchSampler(range(24), batch_size=4, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        # Check the shards when the dataset is not a round multiple of batch size.
        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 21]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [0, 1]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        # Check the shards when the dataset is not a round multiple of batch size or num_processes.
        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 0]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [1, 2]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        # Check the shards when the dataset is very small.
        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=False)
        expected = [[[0, 1]], [[0, 1]]]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=True)
        expected = [[], []]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

    def test_batch_sampler_shards_with_no_splits_no_even(self):
        # Check the shards when the dataset is a round multiple of total batch size.
        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        # Check the shards when the dataset is a round multiple of batch size but not total batch size.
        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        # Check the shards when the dataset is not a round multiple of batch size but has a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        # Check the shards when the dataset is not a round multiple of batch size but and has not a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        # Check the shards when the dataset is very small.
        batch_sampler = BatchSampler(range(2), batch_size=3, drop_last=False)
        expected = [[[0, 1]], []]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

        batch_sampler = BatchSampler(range(2), batch_size=3, drop_last=True)
        expected = [[], []]
        self.check_batch_sampler_shards(batch_sampler, expected, even_batches=False)

    def test_batch_sampler_shards_with_splits_no_even(self):
        # Check the shards when the dataset is a round multiple of batch size.
        batch_sampler = BatchSampler(range(24), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 21]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [22, 23]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        batch_sampler = BatchSampler(range(24), batch_size=4, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        # Check the shards when the dataset is not a round multiple of batch size.
        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 21]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        # Check the shards when the dataset is not a round multiple of batch size or num_processes.
        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]],
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        # Check the shards when the dataset is very small.
        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=False)
        expected = [[[0, 1]], []]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=True)
        expected = [[], []]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True, even_batches=False)

    def test_batch_sampler_with_varying_batch_size(self):
        batch_sampler = [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11], [12, 13]]
        batch_sampler_shards = [BatchSamplerShard(batch_sampler, 2, i, even_batches=False) for i in range(2)]

        assert len(batch_sampler_shards[0]) == 3
        assert len(batch_sampler_shards[1]) == 2

        assert list(batch_sampler_shards[0]) == [[0, 1, 2], [5, 6, 7, 8], [12, 13]]
        assert list(batch_sampler_shards[1]) == [[3, 4], [9, 10, 11]]

    def check_iterable_dataset_shards(
        self, dataset, seed, batch_size, drop_last=False, num_processes=2, split_batches=False
    ):
        random.seed(seed)
        reference = list(dataset)

        iterable_dataset_shards = [
            IterableDatasetShard(
                dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                num_processes=num_processes,
                process_index=i,
                split_batches=split_batches,
            )
            for i in range(num_processes)
        ]
        iterable_dataset_lists = []
        for iterable_dataset_shard in iterable_dataset_shards:
            # Since our random iterable dataset will be... random... we need to use a seed to get reproducible results.
            random.seed(seed)
            iterable_dataset_lists.append(list(iterable_dataset_shard))

        shard_batch_size = batch_size // num_processes if split_batches else batch_size
        # All iterable dataset shard should have the same length, a round multiple of shard_batch_size
        first_list = iterable_dataset_lists[0]
        for l in iterable_dataset_lists[1:]:
            assert len(l) == len(first_list)
            assert (len(l) % shard_batch_size) == 0

        observed = []
        for idx in range(0, len(first_list), shard_batch_size):
            for l in iterable_dataset_lists:
                observed += l[idx : idx + shard_batch_size]

        if not drop_last:
            while len(reference) < len(observed):
                reference += reference
        assert observed == reference[: len(observed)]

    def test_iterable_dataset_shard(self):
        seed = 42
        dataset = RandomIterableDataset()

        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=False, split_batches=False)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=True, split_batches=False)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=False, split_batches=True)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=True, split_batches=True)

        # Edge case with a very small dataset
        dataset = RandomIterableDataset(max_length=2)

        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=False, split_batches=False)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=True, split_batches=False)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=False, split_batches=True)
        self.check_iterable_dataset_shards(dataset, seed, batch_size=4, drop_last=True, split_batches=True)

    def test_iterable_dataset_using_none_batch_size(self):
        dataset = SimpleIterableDataset(100)
        dataloader = DataLoader(dataset, batch_size=None)
        dataloader = prepare_data_loader(dataloader)
        for d in dataloader:
            assert isinstance(d, torch.Tensor)

    @parameterized.expand([1, 2], name_func=parameterized_custom_name_func)
    def test_reproducibility(self, num_processes):
        set_seed(21)
        dataset = list(range(6))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = prepare_data_loader(dataloader, num_processes=num_processes)
        vals_1 = []
        for val in dataloader:
            vals_1.append(val)

        # check same order for same seed
        set_seed(21)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = prepare_data_loader(dataloader, num_processes=num_processes)
        vals_2 = []
        for val in dataloader:
            vals_2.append(val)

        assert vals_1 == vals_2

        # check different order for different seed
        set_seed(42)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = prepare_data_loader(dataloader, num_processes=num_processes)
        vals_3 = []
        for val in dataloader:
            vals_3.append(val)

        assert vals_1 != vals_3

    def test_skip_batch_sampler(self):
        batch_sampler = BatchSampler(range(16), batch_size=4, drop_last=False)
        new_batch_sampler = SkipBatchSampler(batch_sampler, 2)
        assert list(new_batch_sampler) == [[8, 9, 10, 11], [12, 13, 14, 15]]

    def test_dataloader_inheritance(self):
        """
        `DataLoaderAdapter`'s parent classes are dynamically constructed, assert that subclasses of DataLoaderAdapter
        are instances of DataLoader and DataLoaderStateMixin.
        """
        skip_dl = SkipDataLoader(range(16), batch_size=4, skip_batches=2)
        dl_shard = DataLoaderShard(range(16), batch_size=4)
        dl_dispatcher = DataLoaderDispatcher(range(16), batch_size=4)

        # Test dataloaders are instances of instantiated classes
        # These asserts look redundant, but it's worth checking since we are doing magic tricks such as dynamically overriding __class__
        assert isinstance(skip_dl, SkipDataLoader)
        assert isinstance(dl_shard, DataLoaderShard)
        assert isinstance(dl_dispatcher, DataLoaderDispatcher)

        # Test dataloaders are instances of base classes
        assert isinstance(skip_dl, DataLoader)
        assert isinstance(dl_shard, DataLoader)
        assert isinstance(dl_dispatcher, DataLoader)

        assert isinstance(dl_shard, DataLoaderStateMixin)
        assert isinstance(dl_dispatcher, DataLoaderStateMixin)

        assert isinstance(skip_dl.base_dataloader, DataLoader)
        assert isinstance(dl_shard.base_dataloader, DataLoader)
        assert isinstance(dl_dispatcher.base_dataloader, DataLoader)

        with pytest.raises(AttributeError):
            _ = DataLoaderShard.base_dataloader

    def test_skip_data_loader(self):
        dataloader = SkipDataLoader(list(range(16)), batch_size=4, skip_batches=2)
        assert [t.tolist() for t in dataloader] == [[8, 9, 10, 11], [12, 13, 14, 15]]

    def test_skip_first_batches(self):
        dataloader = DataLoader(list(range(16)), batch_size=4)
        new_dataloader = skip_first_batches(dataloader, num_batches=2)
        assert [t.tolist() for t in new_dataloader] == [[8, 9, 10, 11], [12, 13, 14, 15]]

    def test_end_of_dataloader(self):
        dataloader = DataLoaderShard(list(range(16)), batch_size=4)
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

        # Test it also works on the second iteration
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

    def test_end_of_dataloader_dispatcher(self):
        dataloader = DataLoaderDispatcher(range(16), batch_size=4)
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

        # Test it also works on the second iteration
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

    def test_set_epoch_in_batch_sampler(self):
        # Ensure that set_epoch gets propagated to custom batch samplers that accept it
        dataset = list(range(16))
        generator = torch.Generator()
        batch_sampler = SimpleBatchSampler(dataset, batch_size=4, drop_last=False, generator=generator, seed=12)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

        accelerator = Accelerator()
        dataloader = accelerator.prepare_data_loader(dataloader)

        assert batch_sampler.epoch == 0
        dataloader.set_epoch(1)
        assert batch_sampler.epoch == 1

    def test_ensure_dataloader_gets_cleaned_up(self):
        # Ensure that the dataloader gets cleaned up properly
        class Dummy:
            def __init__(self):
                dataset = list(range(16))
                dataloader = DataLoader(dataset, batch_size=4)

                self.accelerator = Accelerator()
                self.dataloader = self.accelerator.prepare_data_loader(dataloader)

                self.iter = iter(self.dataloader)

            def __call__(self, *args, **kwds):
                return next(self.iter)

        instance = Dummy()
        assert instance().tolist() == [0, 1, 2, 3]

        # Create weak references to the objects that *should* be cleaned up if the instance is deleted
        accelerator_ref = weakref.ref(instance.accelerator)
        dataloader_ref = weakref.ref(instance.dataloader)
        gradient_state_ref = weakref.ref(instance.dataloader.gradient_state)

        del instance

        assert accelerator_ref() is None
        assert dataloader_ref() is None
        assert gradient_state_ref() is None


class StatefulDataLoaderTester(AccelerateTestCase):
    @require_torchdata_stateful_dataloader
    def test_skip_data_loader(self):
        dataloader = SkipDataLoader(list(range(16)), batch_size=4, skip_batches=2, use_stateful_dataloader=True)
        assert isinstance(dataloader, StatefulDataLoader)
        assert [t.tolist() for t in dataloader] == [[8, 9, 10, 11], [12, 13, 14, 15]]

    @require_torchdata_stateful_dataloader
    def test_end_of_dataloader(self):
        dataloader = DataLoaderShard(list(range(16)), batch_size=4, use_stateful_dataloader=True)
        assert dataloader.use_stateful_dataloader
        assert isinstance(dataloader, StatefulDataLoader)
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

        # Test it also works on the second iteration
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

    @require_torchdata_stateful_dataloader
    def test_end_of_dataloader_dispatcher(self):
        dataloader = DataLoaderDispatcher(range(16), batch_size=4, use_stateful_dataloader=True)
        assert isinstance(dataloader, StatefulDataLoader)
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

        # Test it also works on the second iteration
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

    @parameterized.expand([0, 2], name_func=parameterized_custom_name_func)
    @require_torchdata_stateful_dataloader
    def test_dataloader_state_dict(self, num_workers):
        """
        Test that saving a stateful dataloader's state, then loading it back, gives the same results.
        """
        dataset = list(range(16))
        dataloader = DataLoaderShard(dataset, batch_size=4, use_stateful_dataloader=True, num_workers=num_workers)

        assert dataloader.use_stateful_dataloader
        assert isinstance(dataloader, StatefulDataLoader)
        vals = []
        for idx, val in enumerate(dataloader):
            vals.append(val)
            if idx == 1:
                sd = dataloader.state_dict()
        assert len(vals) == 4

        dataloader2 = DataLoaderShard(dataset, batch_size=4, use_stateful_dataloader=True, num_workers=num_workers)
        dataloader2.load_state_dict(sd)

        data1 = vals[2:]
        data2 = list(dataloader2)
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert torch.allclose(d1, d2)

    @parameterized.expand([0, 2], name_func=parameterized_custom_name_func)
    @require_torchdata_stateful_dataloader
    def test_dataloader_dispatcher_state_dict(self, num_workers):
        """
        Test that saving a stateful dataloader's state, then loading it back, gives the same results.
        """
        dataset = list(range(16))
        dataloader = DataLoaderDispatcher(dataset, batch_size=4, use_stateful_dataloader=True, num_workers=num_workers)

        assert dataloader.use_stateful_dataloader
        assert isinstance(dataloader, StatefulDataLoader)
        vals = []
        for idx, val in enumerate(dataloader):
            vals.append(val)
            if idx == 1:
                sd = dataloader.state_dict()
        assert len(vals) == 4
        dataloader2 = DataLoaderDispatcher(
            dataset, batch_size=4, use_stateful_dataloader=True, num_workers=num_workers
        )
        dataloader2.load_state_dict(sd)

        data1 = vals[2:]
        data2 = list(dataloader2)
        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert torch.allclose(d1, d2)

    @require_torchdata_stateful_dataloader
    def test_dataloader_inheritance(self):
        """
        `DataLoaderAdapter`'s parent classes are dynamically constructed, assert that if use_stateful_dataloader=True,
        subclasses of DataLoaderAdapter are instances of StatefulDataLoader and DataLoaderStateMixin.
        """
        skip_dl = SkipDataLoader(range(16), batch_size=4, skip_batches=2, use_stateful_dataloader=True)
        dl_shard = DataLoaderShard(range(16), batch_size=4, use_stateful_dataloader=True)
        dl_dispatcher = DataLoaderDispatcher(range(16), batch_size=4, use_stateful_dataloader=True)

        # Test dataloaders are instances of instantiated classes
        # These asserts look redundant, but it's worth checking since we are doing magic tricks such as dynamically overriding __class__
        assert isinstance(skip_dl, SkipDataLoader)
        assert isinstance(dl_shard, DataLoaderShard)
        assert isinstance(dl_dispatcher, DataLoaderDispatcher)

        assert isinstance(skip_dl, StatefulDataLoader)
        assert isinstance(dl_shard, StatefulDataLoader)
        assert isinstance(dl_dispatcher, StatefulDataLoader)

        assert isinstance(dl_shard, DataLoaderStateMixin)
        assert isinstance(dl_dispatcher, DataLoaderStateMixin)

        assert isinstance(skip_dl.base_dataloader, StatefulDataLoader)
        assert isinstance(dl_shard.base_dataloader, StatefulDataLoader)
        assert isinstance(dl_dispatcher.base_dataloader, StatefulDataLoader)

    @parameterized.expand([0, 2], name_func=parameterized_custom_name_func)
    @require_torchdata_stateful_dataloader
    def test_stateful_dataloader_adapter_equivalent_to_torchdata_stateful_dataloader(self, num_workers):
        """
        Assert that `state_dict()` and `load_state_dict()` for derived subclasses of `DataLoaderAdapter` produce
        the same behavior as `state_dict()` and `load_state_dict()` for `StatefulDataLoader`.
        """
        dataset = list(range(64))

        # Set the seed for reproducibility
        def g():
            return torch.Generator().manual_seed(42)

        accelerator = Accelerator()
        stateful_dl = StatefulDataLoader(dataset, batch_size=4, num_workers=num_workers, generator=g())
        skip_dl = SkipDataLoader(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        dl_shard = DataLoaderShard(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        dl_dispatcher = DataLoaderDispatcher(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )

        dataloaders_under_test = [skip_dl, dl_shard, dl_dispatcher]

        num_batches_to_skip = 8

        def get_first_n_batches(dl, n, device):
            """
            Iterate over the first `n` batches of a dataloader then break, returning the batches in a list.
            """
            batches = []
            for idx, batch in enumerate(dl):
                if idx == n - 1:
                    if hasattr(dl, "end"):
                        dl.end()
                    break
                batches.append(batch.to(device))
            return batches

        # Iterate over all of the dataloaders identically, expect the same values
        expected_batches = get_first_n_batches(stateful_dl, num_batches_to_skip, accelerator.device)
        batches_from_dataloaders = [
            get_first_n_batches(dl, num_batches_to_skip, accelerator.device) for dl in dataloaders_under_test
        ]

        for dl_batches in batches_from_dataloaders:
            for expected, actual in zip(expected_batches, dl_batches):
                assert torch.allclose(expected, actual)

        # The adapters should all produce the same state_dict as the reference stateful dataloader
        expected_state_dict = stateful_dl.state_dict()
        skip_dl_state_dict = skip_dl.state_dict()
        dl_shard_state_dict = dl_shard.state_dict()
        dl_dispatcher_state_dict = dl_dispatcher.state_dict()

        assert expected_state_dict == skip_dl_state_dict
        assert expected_state_dict == dl_shard_state_dict
        assert expected_state_dict == dl_dispatcher_state_dict

        # Load the state dict into new dataloaders
        manual_skip_dl = SkipDataLoader(
            dataset,
            batch_size=4,
            num_workers=num_workers,
            generator=g(),
            skip_batches=num_batches_to_skip,
            use_stateful_dataloader=True,
        )
        loaded_stateful_dl = StatefulDataLoader(dataset, batch_size=4, num_workers=num_workers, generator=g())
        loaded_stateful_dl.load_state_dict(expected_state_dict)
        loaded_skip_dl = SkipDataLoader(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_skip_dl.load_state_dict(expected_state_dict)
        loaded_dl_shard = DataLoaderShard(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_dl_shard.load_state_dict(expected_state_dict)
        loaded_dl_dispatcher = DataLoaderDispatcher(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_dl_dispatcher.load_state_dict(expected_state_dict)

        # Continue the iteration, expecting identical behavior across the board
        def get_all_batches(dl, device):
            """
            Iterate over all batches of a dataloader, returning (batches, num_batches_yielded)
            """
            batches = []
            num_batches_yielded = 0
            for batch in dl:
                batches.append(batch.to(device))
                num_batches_yielded += 1
            return (batches, num_batches_yielded)

        expected_batch_results = get_all_batches(loaded_stateful_dl, accelerator.device)
        dataloader_batch_results = [
            get_all_batches(dl, accelerator.device)
            for dl in [manual_skip_dl, loaded_skip_dl, loaded_dl_shard, loaded_dl_dispatcher]
        ]
        for dl_results in dataloader_batch_results:
            for expected, actual in zip(expected_batches, dl_batches):
                assert torch.allclose(expected[0], actual[0])
                assert expected_batch_results[1] == dl_results[1]

        assert accelerator.gradient_state.active_dataloader is None

    @parameterized.expand([0, 2], name_func=parameterized_custom_name_func)
    @require_torchdata_stateful_dataloader
    def test_decoupled_stateful_dataloader_adapter_equivalent_to_torchdata_stateful_dataloader(self, num_workers):
        """
        Assert that `state_dict()` and `load_state_dict()` for derived subclasses of `DataLoaderAdapter` produce
        the same behavior as `state_dict()` and `load_state_dict()` for `StatefulDataLoader` when *not* using
        Accelerator (and instead using the decoupled `PartialState` workflow).
        """
        dataset = list(range(64))

        # Set the seed for reproducibility
        def g():
            return torch.Generator().manual_seed(42)

        state = PartialState()
        stateful_dl = StatefulDataLoader(dataset, batch_size=4, num_workers=num_workers, generator=g())
        skip_dl = SkipDataLoader(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        dl_shard = DataLoaderShard(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        dl_dispatcher = DataLoaderDispatcher(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )

        dataloaders_under_test = [skip_dl, dl_shard, dl_dispatcher]

        num_batches_to_skip = 8

        def get_first_n_batches(dl, n, device):
            """
            Iterate over the first `n` batches of a dataloader then break, returning the batches in a list.
            """
            batches = []
            for idx, batch in enumerate(dl):
                if idx == n - 1:
                    if hasattr(dl, "end"):
                        dl.end()
                    break
                batches.append(batch.to(device))
            return batches

        # Iterate over all of the dataloaders identically, expect the same values
        expected_batches = get_first_n_batches(stateful_dl, num_batches_to_skip, state.device)
        batches_from_dataloaders = [
            get_first_n_batches(dl, num_batches_to_skip, state.device) for dl in dataloaders_under_test
        ]

        for dl_batches in batches_from_dataloaders:
            for expected, actual in zip(expected_batches, dl_batches):
                assert torch.allclose(expected, actual)

        # The adapters should all produce the same state_dict as the reference stateful dataloader
        expected_state_dict = stateful_dl.state_dict()
        skip_dl_state_dict = skip_dl.state_dict()
        dl_shard_state_dict = dl_shard.state_dict()
        dl_dispatcher_state_dict = dl_dispatcher.state_dict()

        assert expected_state_dict == skip_dl_state_dict
        assert expected_state_dict == dl_shard_state_dict
        assert expected_state_dict == dl_dispatcher_state_dict

        # Load the state dict into new dataloaders
        manual_skip_dl = SkipDataLoader(
            dataset,
            batch_size=4,
            num_workers=num_workers,
            generator=g(),
            skip_batches=num_batches_to_skip,
            use_stateful_dataloader=True,
        )
        loaded_stateful_dl = StatefulDataLoader(dataset, batch_size=4, num_workers=num_workers, generator=g())
        loaded_stateful_dl.load_state_dict(expected_state_dict)
        loaded_skip_dl = SkipDataLoader(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_skip_dl.load_state_dict(expected_state_dict)
        loaded_dl_shard = DataLoaderShard(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_dl_shard.load_state_dict(expected_state_dict)
        loaded_dl_dispatcher = DataLoaderDispatcher(
            dataset, batch_size=4, num_workers=num_workers, generator=g(), use_stateful_dataloader=True
        )
        loaded_dl_dispatcher.load_state_dict(expected_state_dict)

        # Continue the iteration, expecting identical behavior across the board
        def get_all_batches(dl, device):
            """
            Iterate over all batches of a dataloader, returning (batches, num_batches_yielded)
            """
            batches = []
            num_batches_yielded = 0
            for batch in dl:
                batches.append(batch.to(device))
                num_batches_yielded += 1
            return (batches, num_batches_yielded)

        expected_batch_results = get_all_batches(loaded_stateful_dl, state.device)
        dataloader_batch_results = [
            get_all_batches(dl, state.device)
            for dl in [manual_skip_dl, loaded_skip_dl, loaded_dl_shard, loaded_dl_dispatcher]
        ]
        for dl_results in dataloader_batch_results:
            for expected, actual in zip(expected_batches, dl_batches):
                assert torch.allclose(expected[0], actual[0])
                assert expected_batch_results[1] == dl_results[1]

        # Using the decoupled (`PartialState`) workflow, GradientState should be automatically initialized (with
        # default parameters) by `DataLoaderDispatcher`
        assert GradientState._shared_state != {}, "GradientState should already be initialized!"

        gradient_state = GradientState()
        assert gradient_state.active_dataloader is None
