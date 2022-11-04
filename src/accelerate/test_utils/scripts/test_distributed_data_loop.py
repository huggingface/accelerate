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

from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate.accelerator import Accelerator


def create_accelerator(even_batches=True):
    accelerator = Accelerator(even_batches=even_batches)
    assert accelerator.num_processes == 2, "this script expects that two GPUs are available"
    return accelerator


def create_dataloader(accelerator: Accelerator, dataset_size: int, batch_size: int):
    """
    Create a simple DataLoader to use during the test cases
    """
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


if __name__ == "__main__":
    accelerator = create_accelerator()

    accelerator.print("Test that even_batches variable ensures uniform batches across processes")
    test_default_ensures_even_batch_sizes()

    accelerator.print("Run tests with even_batches disabled")
    test_can_disable_even_batches()

    accelerator.print("Test joining uneven inputs")
    test_can_join_uneven_inputs()
