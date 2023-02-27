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

import torch

from accelerate import PartialState
from accelerate.utils.operations import broadcast, gather, pad_across_processes, reduce


state = PartialState()


def create_tensor():
    return (torch.arange(state.num_processes) + 1 + (state.num_processes * state.process_index)).to(state.device)


def test_gather():
    tensor = create_tensor()
    gathered_tensor = gather(tensor)
    assert gathered_tensor.shape == torch.Size([state.num_processes**2])
    assert gathered_tensor.tolist() == list(range(1, state.num_processes**2 + 1))


def test_broadcast():
    tensor = create_tensor()
    broadcasted_tensor = broadcast(tensor)
    assert broadcasted_tensor.shape == torch.Size([state.num_processes])
    assert broadcasted_tensor.tolist() == list(range(1, state.num_processes + 1))


def test_pad_across_processes():
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


def test_reduce_sum():
    # For now runs on only two processes
    if state.num_processes != 2:
        return
    tensor = create_tensor()
    reduced_tensor = reduce(tensor, "sum")
    truth_tensor = torch.tensor([4.0, 6]).to(state.device)
    assert torch.allclose(reduced_tensor, truth_tensor), f"{reduced_tensor} != {truth_tensor}"


def test_reduce_mean():
    # For now runs on only two processes
    if state.num_processes != 2:
        return
    tensor = create_tensor()
    reduced_tensor = reduce(tensor, "mean")
    truth_tensor = torch.tensor([2.0, 3]).to(state.device)
    assert torch.allclose(reduced_tensor, truth_tensor), f"{reduced_tensor} != {truth_tensor}"


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def main():
    state.print("testing gather")
    test_gather()
    state.print("testing broadcast")
    test_broadcast()
    state.print("testing pad_across_processes")
    test_pad_across_processes()
    state.print("testing reduce_sum")
    test_reduce_sum()
    state.print("testing reduce_mean")
    test_reduce_mean()


if __name__ == "__main__":
    main()
