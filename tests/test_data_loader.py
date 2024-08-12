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
import unittest

import pytest
from parameterized import parameterized
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from accelerate import Accelerator
from accelerate.data_loader import (
    BatchSamplerShard,
    CustomTypesDataLoader,
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
    SkipBatchSampler,
    SkipDataLoader,
    skip_first_batches,
)
from accelerate.utils import DataLoaderConfiguration

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


class DataLoaderTester(unittest.TestCase):
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

    def test_skip_batch_sampler(self):
        batch_sampler = BatchSampler(range(16), batch_size=4, drop_last=False)
        new_batch_sampler = SkipBatchSampler(batch_sampler, 2)
        assert list(new_batch_sampler) == [[8, 9, 10, 11], [12, 13, 14, 15]]

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
        Accelerator()
        dataloader = DataLoaderDispatcher(range(16), batch_size=4)
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

        # Test it also works on the second iteration
        for idx, _ in enumerate(dataloader):
            assert dataloader.end_of_dataloader == (idx == 3)

    @staticmethod
    def _get_custom_iterable(data):
        class MyCustomType:
            def __init__(self):
                self.data = data
            def __iter__(self):
                return iter(self.data)
        return MyCustomType()

    @staticmethod
    def check_custom_types_iterable(dataloader, expected_batches, device):
        assert isinstance(dataloader, CustomTypesDataLoader)
        assert len(expected_batches) == len(list(dataloader))
        for _ in range(2):
            for batch, expected_batch in zip(dataloader, expected_batches):
                # And that each time we get the expected tensor on the device we specified
                assert batch.tolist() == expected_batch
                assert batch.device.type == device

    @parameterized.expand([
        ("nested under dataloader wrapper", True),
        ("without nested dataloader wrapper", False),
    ])
    def test_custom_types_dataloader(self, _, wrap_with_dataloader):
        device = "cuda"
        custom_iterable = self._get_custom_iterable(data=list(range(8)))
        if wrap_with_dataloader:
            custom_iterable = DataLoader(custom_iterable, batch_size=4)
            kwargs = {}
        else:
            kwargs = {"batch_size": 4}
        dataloader = CustomTypesDataLoader(custom_iterable, device=device, **kwargs)
        expected_batches = [[0,1,2,3], [4,5,6,7]]
        self.check_custom_types_iterable(dataloader, expected_batches, device)

    @parameterized.expand([
        ("nested under dataloader wrapper", True),
        ("without nested dataloader wrapper", False),
    ])
    def test_custom_types_via_prepare(self, _, wrap_with_dataloader):
        device = "cuda"
        batch_size = 4
        dataloader_config = DataLoaderConfiguration(custom_types=True)
        custom_iterable = self._get_custom_iterable(data=list(range(8)))
        if wrap_with_dataloader:
            # If it's a data loader, we pull the batch size off the dataloader
            custom_iterable = DataLoader(custom_iterable, batch_size=batch_size)
        else:
            # Otherwise we need to specify it through the dataloader config
            dataloader_config.custom_type_batch_size=batch_size
        accelerator = Accelerator(dataloader_config=dataloader_config)
        dataloader = accelerator.prepare(custom_iterable)
        expected_batches = [[0,1,2,3], [4,5,6,7]]
        self.check_custom_types_iterable(dataloader, expected_batches, device)

    def test_prepare_custom_types_dataloader_is_idempotent(self):
        accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(custom_types=True))
        custom_iterable = DataLoader(self._get_custom_iterable(data=list(range(8))), batch_size=4)
        dataloader = CustomTypesDataLoader(custom_iterable)
        prepared_dataloader = accelerator.prepare(dataloader)
        assert isinstance(prepared_dataloader, CustomTypesDataLoader)
        assert dataloader.dataset == prepared_dataloader.dataset

    def test_prepare_custom_types_dataloader_conflicting_batch_sizes(self):
        # Ensure we can't pass a batch size for custom types and a wrapped
        # dataloader unless the batch sizes are the same value
        accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(custom_types=True, custom_type_batch_size=2))
        dataloader = DataLoader(self._get_custom_iterable(data=list(range(8))), batch_size=4)
        with pytest.raises(ValueError):
            accelerator.prepare(dataloader)
