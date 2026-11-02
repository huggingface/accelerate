# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader, skip_first_batches


class VariableBatchSampler(BatchSampler):
    def __init__(self, drop_last=False):
        self.batch_size = None
        self.drop_last = drop_last
        self.sampler = range(6)

    def __len__(self):
        return 4

    def __iter__(self):
        yield from ([0, 1], [2, 3], [4], [5])


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("skip", [0, 1])
@pytest.mark.parametrize("drop_last", [False, True])
def test_variable_metadata_and_skip(rank, skip, drop_last):
    loader = prepare_data_loader(
        DataLoader(range(6), batch_sampler=VariableBatchSampler(drop_last)),
        num_processes=2,
        process_index=rank,
        even_batches=False,
        split_batches=False,
    )
    if skip:
        loader = skip_first_batches(loader, skip)
    assert loader.total_batch_size is None
    assert len(loader) == 2 - skip
    expected = [[0, 1], [4]] if rank == 0 else [[2, 3], [5]]
    actual = []
    for batch in loader:
        assert loader.remainder == -1
        actual.append(batch.tolist())
    assert actual == expected[skip:]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("skip", [0, 1])
def test_uniform_metadata_and_remainder_unchanged(split, skip):
    loader = prepare_data_loader(
        DataLoader(range(10), batch_size=4),
        num_processes=2,
        process_index=0,
        split_batches=split,
    )
    if skip:
        loader = skip_first_batches(loader, skip)
    assert loader.total_batch_size == (4 if split else 8)
    for _ in loader:
        assert loader.remainder == 2


def test_unknown_remainder_does_not_trim_metrics():
    accelerator = Accelerator(cpu=True)
    loader = prepare_data_loader(
        DataLoader(range(6), batch_sampler=VariableBatchSampler()),
        num_processes=1,
        process_index=0,
        even_batches=False,
    )
    seen = []
    for batch in loader:
        assert loader.remainder == -1
        gathered = accelerator.gather_for_metrics(batch)
        torch.testing.assert_close(gathered, batch)
        seen.extend(gathered.tolist())
    assert seen == list(range(6))
