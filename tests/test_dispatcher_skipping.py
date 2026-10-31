# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import itertools

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from accelerate import Accelerator, debug_launcher
from accelerate.data_loader import skip_first_batches
from accelerate.test_utils import require_cpu
from accelerate.utils import DataLoaderConfiguration


class SizedStream(IterableDataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return iter(range(self.size))

    def __len__(self):
        return self.size


def collate_nested(samples):
    return {"values": torch.tensor(samples), "nested": [torch.tensor(samples) + 100]}


def check_dispatched_skipping(split_batches):
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=split_batches),
    )
    assert accelerator.num_processes == 2
    for streaming, size, drop_last in itertools.product([False, True], [8, 9, 14, 16, 17, 32], [False, True]):

        def make_loader(streaming=streaming, size=size, drop_last=drop_last):
            dataset = SizedStream(size) if streaming else torch.arange(size)
            return accelerator.prepare(DataLoader(dataset, batch_size=4, drop_last=drop_last))

        expected = list(make_loader())
        for skipped in sorted({0, 1, 2, len(expected), len(expected) + 1}):
            loader = skip_first_batches(make_loader(), skipped)
            torch.testing.assert_close(list(loader), expected[skipped:])
            assert len(loader) == len(expected[skipped:])
            assert accelerator.gradient_state.active_dataloader is None

        loader = skip_first_batches(skip_first_batches(make_loader(), 1), 1)
        torch.testing.assert_close(list(loader), expected[2:])
        assert len(loader) == len(expected[2:])
    # Keep custom batching/collation and worker-backed loading intact as well.
    for workers in [0, 1]:

        def make_custom_loader(workers=workers):
            return accelerator.prepare(
                DataLoader(
                    list(range(34)),
                    batch_sampler=BatchSampler(range(34), batch_size=4, drop_last=False),
                    collate_fn=collate_nested,
                    num_workers=workers,
                )
            )

        expected = list(make_custom_loader())
        loader = skip_first_batches(make_custom_loader(), 1)
        torch.testing.assert_close(list(loader), expected[1:])
        assert len(loader) == len(expected[1:])
    accelerator.end_training()


@require_cpu
def test_dispatched_skipping_preserves_grouping():
    debug_launcher(check_dispatched_skipping, args=(False,))


@require_cpu
def test_split_dispatched_skipping_preserves_padding():
    debug_launcher(check_dispatched_skipping, args=(True,))
