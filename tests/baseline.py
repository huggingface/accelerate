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

import torch

from accelerate import PartialState, Accelerator
from accelerate.test_utils.testing import assert_exception
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import (
    DistributedOperationException,
    broadcast,
    copy_tensor_to_devices,
    gather,
    gather_object,
    pad_across_processes,
    reduce,
)


def create_tensor(state):
    return (torch.arange(state.num_processes) + 1.0 + (state.num_processes * state.process_index)).to(state.device)


def test_gather(state):
    tensor = create_tensor(state)
    gathered_tensor = gather(tensor)
    assert gathered_tensor.tolist() == list(range(1, state.num_processes**2 + 1))


def test_gather_object(state):
    # Gather objects in TorchXLA is not supported.
    if state.distributed_type == DistributedType.XLA:
        return
    obj = [state.process_index]
    gathered_obj = gather_object(obj)
    assert len(gathered_obj) == state.num_processes, f"{gathered_obj}, {len(gathered_obj)} != {state.num_processes}"
    assert gathered_obj == list(range(state.num_processes)), f"{gathered_obj} != {list(range(state.num_processes))}"


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.local_process_index == 0:
        print("**Initialization**")
    state.wait_for_everyone()

    if state.distributed_type == DistributedType.MULTI_GPU:
        num_processes_per_node = torch.cuda.device_count()
    else:
        num_processes_per_node = state.num_processes

    # We only run this test on non-multinode
    if state.process_index == 0:
        print("\n**Test gather operation**")
    test_gather(state)
    if state.process_index == 0:
        print("\n**Test gather_object operation**")
    test_gather_object(state)



if __name__ == "__main__":
    main()
