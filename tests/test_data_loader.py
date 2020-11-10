import unittest

from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard

class DataLoaderTester(unittest.TestCase):
    def check_batch_sampler_shards(self, batch_sampler, expected, split_batches=False):
        batch_sampler_shards = [BatchSamplerShard(batch_sampler, 2, i, split_batches) for i in range(2)]
        batch_sampler_lists = [list(batch_sampler_shard) for batch_sampler_shard in batch_sampler_shards]
        self.assertListEqual(batch_sampler_lists, expected)

    def test_batch_sampler_shards_with_no_splits(self):
        # Check the shards when the dataset is a round multiple of total batch size.
        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(24), batch_size=3, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is a round multiple of batch size but not total batch size.
        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [0, 1, 2]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(21), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is not a round multiple of batch size but has a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 0, 1]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(22), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        # Check the shards when the dataset is not a round multiple of batch size but and has not a multiple of
        # num_processes batch.
        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=False)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 0]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17], [1, 2, 3]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected)

        batch_sampler = BatchSampler(range(20), batch_size=3, drop_last=True)
        expected = [
            [[0, 1, 2], [6, 7, 8], [12, 13, 14]],
            [[3, 4, 5], [9, 10, 11], [15, 16, 17]]
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
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [22, 23]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected, split_batches=True)

        batch_sampler = BatchSampler(range(24), batch_size=4, drop_last=True)
        # Expected shouldn't change
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)


        # Check the shards when the dataset is not a round multiple of batch size.
        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 21]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [0, 1]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)

        batch_sampler = BatchSampler(range(22), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)

        # Check the shards when the dataset is not a round multiple of batch size or num_processes.
        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=False)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17], [20, 0]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [1, 2]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)

        batch_sampler = BatchSampler(range(21), batch_size=4, drop_last=True)
        expected = [
            [[0, 1], [4, 5], [8, 9], [12, 13], [16, 17]],
            [[2, 3], [6, 7], [10, 11], [14, 15], [18, 19]]
        ]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)

        # Check the shards when the dataset is very small.
        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=False)
        expected = [[[0, 1]], [[0, 1]]]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)

        batch_sampler = BatchSampler(range(2), batch_size=4, drop_last=True)
        expected = [[], []]
        self.check_batch_sampler_shards(batch_sampler, expected,  split_batches=True)