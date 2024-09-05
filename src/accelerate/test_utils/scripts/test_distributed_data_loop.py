#!/usr/bin/env python

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

import pickle
import tempfile
import warnings
from typing import List
from unittest.mock import Mock

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    TensorDataset,
    default_collate,
)

from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType


NUM_ELEMENTS = 22
NUM_WORKERS = 4
BATCH_SIZE = 4


class DummyDataset(Dataset):
    def __len__(self):
        return NUM_ELEMENTS

    def __getitem__(self, index):
        squeeze = False

        if isinstance(index, int):
            index = [index]
            squeeze = True
        elif isinstance(index, slice):
            index = list(range(*index.indices(self.size)))
        else:
            index = list(index)

        batch = [{"index": i, "label": i % 2, "random_augmentation": torch.rand(1).item()} for i in index]

        if squeeze:
            batch = batch[0]

        return batch


class DummyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data


def create_accelerator(even_batches=True):
    dataloader_config = DataLoaderConfiguration(even_batches=even_batches)
    accelerator = Accelerator(dataloader_config=dataloader_config)
    assert accelerator.num_processes == 2, "this script expects that two GPUs are available"
    return accelerator


def create_dataloader(
    accelerator: Accelerator, dataset_size: int, batch_size: int, iterable: bool = False, shuffle: bool = False
):
    """
    Create a simple DataLoader to use during the test cases
    """
    values = torch.as_tensor(range(dataset_size))
    if shuffle:
        values = values[torch.randperm(values.size(0))]
    if iterable:
        dataset = DummyIterableDataset(values)
    else:
        dataset = TensorDataset(torch.as_tensor(range(dataset_size)))

    dl = DataLoader(dataset, batch_size=batch_size)
    dl = accelerator.prepare(dl)

    return dl


def verify_dataloader_batch_sizes(
    accelerator: Accelerator,
    dataset_size: int,
    batch_size: int,
    process_0_expected_batch_sizes: List[int],
    process_1_expected_batch_sizes: List[int],
):
    """
    A helper function for verifying the batch sizes coming from a prepared dataloader in each process
    """
    dl = create_dataloader(accelerator=accelerator, dataset_size=dataset_size, batch_size=batch_size)

    batch_sizes = [len(batch[0]) for batch in dl]

    if accelerator.process_index == 0:
        assert batch_sizes == process_0_expected_batch_sizes
    elif accelerator.process_index == 1:
        assert batch_sizes == process_1_expected_batch_sizes


def test_default_ensures_even_batch_sizes():
    accelerator = create_accelerator()

    # without padding, we would expect a different number of batches
    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=3,
        batch_size=1,
        process_0_expected_batch_sizes=[1, 1],
        process_1_expected_batch_sizes=[1, 1],
    )

    # without padding, we would expect the same number of batches, but different sizes
    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=7,
        batch_size=2,
        process_0_expected_batch_sizes=[2, 2],
        process_1_expected_batch_sizes=[2, 2],
    )


def test_can_disable_even_batches():
    accelerator = create_accelerator(even_batches=False)

    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=3,
        batch_size=1,
        process_0_expected_batch_sizes=[1, 1],
        process_1_expected_batch_sizes=[1],
    )

    verify_dataloader_batch_sizes(
        accelerator,
        dataset_size=7,
        batch_size=2,
        process_0_expected_batch_sizes=[2, 2],
        process_1_expected_batch_sizes=[2, 1],
    )


def test_can_join_uneven_inputs():
    accelerator = create_accelerator(even_batches=False)

    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)

    dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    batch_idxs = []
    with accelerator.join_uneven_inputs([ddp_model]):
        for batch_idx, batch in enumerate(dl):
            output = ddp_model(batch[0].float())
            loss = output.sum()
            loss.backward()
            batch_idxs.append(batch_idx)

    accelerator.wait_for_everyone()

    if accelerator.process_index == 0:
        assert batch_idxs == [0, 1]
    elif accelerator.process_index == 1:
        assert batch_idxs == [0]


def test_join_raises_warning_for_non_ddp_distributed(accelerator):
    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([Mock()]):
            pass

        assert issubclass(w[-1].category, UserWarning)
        assert "only supported for multi-GPU" in str(w[-1].message)


def test_join_can_override_even_batches():
    default_even_batches = True
    overridden_even_batches = False
    accelerator = create_accelerator(even_batches=default_even_batches)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    train_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)
    valid_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    with accelerator.join_uneven_inputs([ddp_model], even_batches=overridden_even_batches):
        train_dl_overridden_value = train_dl.batch_sampler.even_batches
        valid_dl_overridden_value = valid_dl.batch_sampler.even_batches

    assert train_dl_overridden_value == overridden_even_batches
    assert valid_dl_overridden_value == overridden_even_batches
    assert train_dl.batch_sampler.even_batches == default_even_batches
    assert valid_dl.batch_sampler.even_batches == default_even_batches


def test_join_can_override_for_mixed_type_dataloaders():
    default_even_batches = True
    overridden_even_batches = False
    accelerator = create_accelerator(even_batches=default_even_batches)
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    create_dataloader(accelerator, dataset_size=3, batch_size=1, iterable=True)
    batch_dl = create_dataloader(accelerator, dataset_size=3, batch_size=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            with accelerator.join_uneven_inputs([ddp_model], even_batches=overridden_even_batches):
                batch_dl_overridden_value = batch_dl.batch_sampler.even_batches
        except AttributeError:
            # ensure attribute error is not raised when processing iterable dl
            raise AssertionError

    assert batch_dl_overridden_value == overridden_even_batches
    assert batch_dl.batch_sampler.even_batches == default_even_batches


def test_join_raises_warning_for_iterable_when_overriding_even_batches():
    accelerator = create_accelerator()
    model = torch.nn.Linear(1, 1)
    ddp_model = accelerator.prepare(model)
    create_dataloader(accelerator, dataset_size=3, batch_size=1, iterable=True)

    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
            pass

        assert issubclass(w[-1].category, UserWarning)
        assert "only supported for map-style datasets" in str(w[-1].message)


def test_pickle_accelerator():
    accelerator = create_accelerator()
    data_loader = create_dataloader(accelerator, dataset_size=32, batch_size=4)
    _ = accelerator.prepare(data_loader)
    pickled_accelerator = pickle.dumps(accelerator)
    unpickled_accelerator = pickle.loads(pickled_accelerator)
    # TODO: Maybe this should be implemented as __eq__ for AcceleratorState?
    assert accelerator.state.__dict__ == unpickled_accelerator.state.__dict__


def test_data_loader(data_loader, accelerator):
    # Prepare the DataLoader
    data_loader = accelerator.prepare(data_loader)

    all_examples = []
    for i, batch in enumerate(data_loader):
        index, _ = accelerator.gather_for_metrics((batch["index"], batch["label"]))
        all_examples.extend(index.detach().cpu().numpy().tolist())

    # Sort the examples
    sorted_all_examples = sorted(all_examples)

    # Check if all elements are present in the sorted list of iterated samples
    assert (
        len(set(sorted_all_examples)) == NUM_ELEMENTS
    ), "Not all the dataset elements have been iterated in an epoch due to duplication of samples across processes."


def test_stateful_dataloader(accelerator):
    """
    Tests that a stateful dataloader can be iterated over, saved after a few batches using `load_state_dict`, and then
    resumed from the saved state.

    The result should be the same as the rest of the data that iterated over after saving.
    """
    old_dataloader_config = accelerator.dataloader_config
    try:
        accelerator.dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)
        prepared_dl = create_dataloader(
            accelerator, dataset_size=32 * accelerator.num_processes, batch_size=4, iterable=True, shuffle=True
        )
        untrained_batches = []
        # Calculate what step that will be
        total_batches = 32 * accelerator.num_processes // (4 * accelerator.num_processes)
        last_batch_num = total_batches - 1
        for step, batch in enumerate(prepared_dl):
            # Step just before
            if step == last_batch_num - 1:
                state_dict = prepared_dl.state_dict()
            if step >= last_batch_num:
                # Otherwise grab the "unseen" batches
                untrained_batches.append(batch)
        not_skipped_batches = accelerator.gather(untrained_batches)
        prepared_dl.load_state_dict(state_dict)
        resumed_batches = []
        for batch in prepared_dl:
            resumed_batches.append(batch)
        resumed_batches = accelerator.gather(resumed_batches)
        for b1, b2 in zip(not_skipped_batches, resumed_batches):
            for v1, v2 in zip(b1, b2):
                assert torch.equal(v1, v2), f"Batch {b1} and {b2} are not equal"
    finally:
        accelerator.dataloader_config = old_dataloader_config


def test_stateful_dataloader_save_state(accelerator):
    """
    Tests that a stateful dataloader can be iterated over, saved after a few batches using `Accelerator.save_state`,
    and then resumed from the saved state.

    The result should be the same as the rest of the data that iterated over after saving.
    """
    old_dataloader_config = accelerator.dataloader_config
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            accelerator.dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)
            prepared_dl = create_dataloader(
                accelerator, dataset_size=32 * accelerator.num_processes, batch_size=4, iterable=True, shuffle=True
            )
            untrained_batches = []
            # Calculate what step that will be
            total_batches = 32 * accelerator.num_processes // (4 * accelerator.num_processes)
            last_batch_num = total_batches - 1
            for step, batch in enumerate(prepared_dl):
                # Step just before
                if step == last_batch_num - 1:
                    accelerator.save_state(tmpdir)
                if step >= last_batch_num:
                    # Otherwise grab the "unseen" batches
                    untrained_batches.append(batch)
            not_skipped_batches = accelerator.gather(untrained_batches)
            accelerator.load_state(tmpdir)
            resumed_batches = []
            for batch in prepared_dl:
                resumed_batches.append(batch)
            resumed_batches = accelerator.gather(resumed_batches)
            for b1, b2 in zip(not_skipped_batches, resumed_batches):
                for v1, v2 in zip(b1, b2):
                    assert torch.equal(v1, v2), f"Batch {b1} and {b2} are not equal"
    finally:
        accelerator.dataloader_config = old_dataloader_config


def main():
    accelerator = create_accelerator()
    torch.manual_seed(accelerator.process_index)

    accelerator.print("Test that even_batches variable ensures uniform batches across processes")
    test_default_ensures_even_batch_sizes()

    accelerator.print("Run tests with even_batches disabled")
    test_can_disable_even_batches()

    accelerator.print("Test joining uneven inputs")
    test_can_join_uneven_inputs()

    accelerator.print("Test overriding even_batches when joining uneven inputs")
    test_join_can_override_even_batches()

    accelerator.print("Test overriding even_batches for mixed dataloader types")
    test_join_can_override_for_mixed_type_dataloaders()

    accelerator.print("Test overriding even_batches raises a warning for iterable dataloaders")
    test_join_raises_warning_for_iterable_when_overriding_even_batches()

    accelerator.print("Test join with non DDP distributed raises warning")
    original_state = accelerator.state.distributed_type
    accelerator.state.distributed_type = DistributedType.FSDP
    test_join_raises_warning_for_non_ddp_distributed(accelerator)
    accelerator.state.distributed_type = original_state

    accelerator.print("Test pickling an accelerator")
    test_pickle_accelerator()

    dataset = DummyDataset()
    # Conventional Dataloader with shuffle=False
    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_data_loader(loader, accelerator)

    # Conventional Dataloader with shuffle=True
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_data_loader(loader, accelerator)

    # Dataloader with batch_sampler
    sampler = BatchSampler(RandomSampler(dataset), batch_size=BATCH_SIZE, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=NUM_WORKERS)
    test_data_loader(loader, accelerator)

    # Dataloader with sampler as an instance of `BatchSampler`
    sampler = BatchSampler(RandomSampler(dataset), batch_size=BATCH_SIZE, drop_last=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None, collate_fn=default_collate, num_workers=NUM_WORKERS)
    test_data_loader(loader, accelerator)
    test_stateful_dataloader(accelerator)
    test_stateful_dataloader_save_state(accelerator)

    accelerator.end_training()


if __name__ == "__main__":
    main()
