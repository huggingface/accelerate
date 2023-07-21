#!/usr/bin/env python

# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import logging
import sys
import torch
from torch.utils.data import IterableDataset, DataLoader

from accelerate import PartialState, Accelerator
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.utils.operations import broadcast, gather, gather_object, pad_across_processes, reduce
import logging

class ListHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super(ListHandler, self).__init__(*args, **kwargs)
        self.logs = []

    def emit(self, record):
        self.logs.append(record)

def create_tensor(state):
    return (torch.arange(state.num_processes) + 1.0 + (state.num_processes * state.process_index)).to(state.device)


def test_gather(state):
    tensor = create_tensor(state)
    gathered_tensor = gather(tensor)
    assert gathered_tensor.tolist() == list(range(1, state.num_processes**2 + 1))


def test_gather_object(state):
    obj = [state.process_index]
    gathered_obj = gather_object(obj)
    assert len(gathered_obj) == state.num_processes, f"{gathered_obj}, {len(gathered_obj)} != {state.num_processes}"
    assert gathered_obj == list(range(state.num_processes)), f"{gathered_obj} != {list(range(state.num_processes))}"


def test_broadcast(state):
    tensor = create_tensor(state)
    broadcasted_tensor = broadcast(tensor)
    assert broadcasted_tensor.shape == torch.Size([state.num_processes])
    assert broadcasted_tensor.tolist() == list(range(1, state.num_processes + 1))


def test_pad_across_processes(state):
    # We need to pad the tensor with one more element if we are the main process
    # to ensure that we can pad
    if state.is_main_process:
        tensor = torch.arange(state.num_processes + 1).to(state.device)
    else:
        tensor = torch.arange(state.num_processes).to(state.device)
    padded_tensor = pad_across_processes(tensor)
    assert padded_tensor.shape == torch.Size([state.num_processes + 1])
    if not state.is_main_process:
        assert padded_tensor.tolist() == list(range(0, state.num_processes)) + [0]


def test_reduce_sum(state):
    # For now runs on only two processes
    if state.num_processes != 2:
        return
    tensor = create_tensor(state)
    reduced_tensor = reduce(tensor, "sum")
    truth_tensor = torch.tensor([4.0, 6]).to(state.device)
    assert torch.allclose(reduced_tensor, truth_tensor), f"{reduced_tensor} != {truth_tensor}"


def test_reduce_mean(state):
    # For now runs on only two processes
    if state.num_processes != 2:
        return
    tensor = create_tensor(state)
    reduced_tensor = reduce(tensor, "mean")
    truth_tensor = torch.tensor([2.0, 3]).to(state.device)
    assert torch.allclose(reduced_tensor, truth_tensor), f"{reduced_tensor} != {truth_tensor}"

def test_gather_for_metrics_with_iterable_dataset(state):
    assert state.num_processes == 2

    class DummyIterableDataset(IterableDataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            for element in self.data:
                yield element

    iterable_dataset = DummyIterableDataset(torch.as_tensor(range(15)))
    dataloader = DataLoader(iterable_dataset, batch_size=4)

    accelerator = Accelerator()
    prepared_dataloader = accelerator.prepare(dataloader)

    assert type(prepared_dataloader) == DataLoaderDispatcher

    if state.is_main_process:
        logger = logging.root.manager.loggerDict['accelerate.accelerator']
        list_handler = ListHandler()
        logger.addHandler(list_handler)

    batches_for_metrics = []
    for _, batch in enumerate(prepared_dataloader):
        print(accelerator.gradient_state.remainder)
        batches_for_metrics.append(accelerator.gather_for_metrics(batch))

    assert torch.cat(batches_for_metrics).size(0) == 15

    if state.is_main_process:
        # inverse assertion
        assert len(list_handler.logs) == 1
        assert "The used dataset had no length, returning gathered tensors." in list_handler.logs[0].msg

        logger.removeHandler(list_handler)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def main():
    state = PartialState()
    state.print(f"State: {state}")
    state.print("testing gather")
    test_gather(state)
    state.print("testing gather_object")
    test_gather_object(state)
    state.print("testing broadcast")
    test_broadcast(state)
    state.print("testing pad_across_processes")
    test_pad_across_processes(state)
    state.print("testing reduce_sum")
    test_reduce_sum(state)
    state.print("testing reduce_mean")
    test_reduce_mean(state)
    state.print("test_gather_for_metrics_with_iterable_dataset")
    test_gather_for_metrics_with_iterable_dataset(state)


if __name__ == "__main__":
    main()
