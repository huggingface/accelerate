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
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.state import AcceleratorState, DistributedType
from accelerate.test_utils import RegressionDataset, RegressionModel, are_the_same_tensors
from accelerate.utils import gather, set_seed, synchronize_rng_states
from packaging import version


def init_state_check():
    # Test we can instantiate this twice in a row.
    state = AcceleratorState()
    if state.local_process_index == 0:
        print("Testing, testing. 1, 2, 3.")
    print(state)


def rng_sync_check():
    state = AcceleratorState()
    synchronize_rng_states(["torch"])
    assert are_the_same_tensors(torch.get_rng_state())
    if state.distributed_type == DistributedType.MULTI_GPU:
        synchronize_rng_states(["cuda"])
        assert are_the_same_tensors(torch.cuda.get_rng_state())
    if version.parse(torch.__version__) >= version.parse("1.6.0"):
        generator = torch.Generator()
        synchronize_rng_states(["generator"], generator=generator)
        assert are_the_same_tensors(generator.get_state())

    if state.local_process_index == 0:
        print("All rng are properly synched.")


def dl_preparation_check():
    state = AcceleratorState()
    length = 32 * state.num_processes

    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    assert torch.equal(result.cpu(), torch.arange(0, length).long())

    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(
        dl,
        state.device,
        state.num_processes,
        state.process_index,
        put_on_device=True,
        split_batches=True,
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    assert torch.equal(result.cpu(), torch.arange(0, length).long())

    if state.process_index == 0:
        print("Non-shuffled dataloader passing.")

    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length))

    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(
        dl,
        state.device,
        state.num_processes,
        state.process_index,
        put_on_device=True,
        split_batches=True,
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length))

    if state.local_process_index == 0:
        print("Shuffled dataloader passing.")


def mock_training(length, batch_size, generator):
    set_seed(42)
    generator.manual_seed(42)
    train_set = RegressionDataset(length=length)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            loss.backward()
            optimizer.step()
    return train_set, model


def training_check():
    state = AcceleratorState()
    generator = torch.Generator()
    batch_size = 8
    length = batch_size * 4 * state.num_processes

    train_set, old_model = mock_training(length, batch_size * state.num_processes, generator)
    assert are_the_same_tensors(old_model.a)
    assert are_the_same_tensors(old_model.b)

    accelerator = Accelerator()
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    generator.manual_seed(42)
    for epoch in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            accelerator.backward(loss)
            optimizer.step()

    model = accelerator.unwrap_model(model).cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)

    accelerator.print("Training yielded the same results on one CPU or distributed setup with no batch split.")

    accelerator = Accelerator(split_batches=True)
    train_dl = DataLoader(train_set, batch_size=batch_size * state.num_processes, shuffle=True, generator=generator)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    generator.manual_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            accelerator.backward(loss)
            optimizer.step()

    model = accelerator.unwrap_model(model).cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)

    accelerator.print("Training yielded the same results on one CPU or distributes setup with batch split.")

    # Mostly a test that FP16 doesn't crash as the operation inside the model is not converted to FP16
    accelerator = Accelerator(fp16=True)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    generator.manual_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            accelerator.backward(loss)
            optimizer.step()

    model = accelerator.unwrap_model(model).cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.local_process_index == 0:
        print("**Initialization**")
    init_state_check()

    if state.local_process_index == 0:
        print("\n**Test random number generator synchronization**")
    rng_sync_check()

    if state.local_process_index == 0:
        print("\n**DataLoader integration test**")
    dl_preparation_check()

    if state.local_process_index == 0:
        print("\n**Training integration test**")
    training_check()


if __name__ == "__main__":
    main()
