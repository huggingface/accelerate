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

import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
    DistributedType,
    gather,
    is_bf16_available,
    is_ipex_available,
    is_npu_available,
    is_xpu_available,
    set_seed,
    synchronize_rng_states,
)


# TODO: remove RegressionModel4XPU once ccl support empty buffer in broadcasting.
if is_xpu_available():
    from accelerate.test_utils import RegressionModel4XPU as RegressionModel
else:
    from accelerate.test_utils import RegressionModel


def print_main(state):
    print(f"Printing from the main process {state.process_index}")


def print_local_main(state):
    print(f"Printing from the local main process {state.local_process_index}")


def print_last(state):
    print(f"Printing from the last process {state.process_index}")


def print_on(state, process_idx):
    print(f"Printing from process {process_idx}: {state.process_index}")


def process_execution_check():
    accelerator = Accelerator()
    num_processes = accelerator.num_processes
    # Test main_process_first context manager
    path = Path("check_main_process_first.txt")
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            time.sleep(0.1)  # ensure main process takes longest
            with open(path, "a+") as f:
                f.write("Currently in the main process\n")
        else:
            with open(path, "a+") as f:
                f.write("Now on another process\n")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(path, "r") as f:
            text = "".join(f.readlines())
        try:
            assert text.startswith("Currently in the main process\n"), "Main process was not first"
            if num_processes > 1:
                assert text.endswith("Now on another process\n"), "Main process was not first"
            assert (
                text.count("Now on another process\n") == accelerator.num_processes - 1
            ), f"Only wrote to file {text.count('Now on another process') + 1} times, not {accelerator.num_processes}"
        except AssertionError:
            path.unlink()
            raise

    if accelerator.is_main_process and path.exists():
        path.unlink()
    accelerator.wait_for_everyone()
    # Test the decorators
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        accelerator.on_main_process(print_main)(accelerator.state)
    result = f.getvalue().rstrip()
    if accelerator.is_main_process:
        assert result == "Printing from the main process 0", f"{result} != Printing from the main process 0"
    else:
        assert f.getvalue().rstrip() == "", f'{result} != ""'
    f.truncate(0)
    f.seek(0)

    with contextlib.redirect_stdout(f):
        accelerator.on_local_main_process(print_local_main)(accelerator.state)
    if accelerator.is_local_main_process:
        assert f.getvalue().rstrip() == "Printing from the local main process 0"
    else:
        assert f.getvalue().rstrip() == ""
    f.truncate(0)
    f.seek(0)

    with contextlib.redirect_stdout(f):
        accelerator.on_last_process(print_last)(accelerator.state)
    if accelerator.is_last_process:
        assert f.getvalue().rstrip() == f"Printing from the last process {accelerator.state.num_processes - 1}"
    else:
        assert f.getvalue().rstrip() == ""
    f.truncate(0)
    f.seek(0)

    for process_idx in range(num_processes):
        with contextlib.redirect_stdout(f):
            accelerator.on_process(print_on, process_index=process_idx)(accelerator.state, process_idx)
        if accelerator.process_index == process_idx:
            assert f.getvalue().rstrip() == f"Printing from process {process_idx}: {accelerator.process_index}"
        else:
            assert f.getvalue().rstrip() == ""
        f.truncate(0)
        f.seek(0)


def init_state_check():
    # Test we can instantiate this twice in a row.
    state = AcceleratorState()
    if state.local_process_index == 0:
        print("Testing, testing. 1, 2, 3.")
    print(state)


def rng_sync_check():
    state = AcceleratorState()
    synchronize_rng_states(["torch"])
    assert are_the_same_tensors(torch.get_rng_state()), "RNG states improperly synchronized on CPU."
    if state.distributed_type == DistributedType.MULTI_GPU:
        synchronize_rng_states(["cuda"])
        assert are_the_same_tensors(torch.cuda.get_rng_state()), "RNG states improperly synchronized on GPU."
    elif state.distributed_type == DistributedType.MULTI_XPU:
        synchronize_rng_states(["xpu"])
        assert are_the_same_tensors(torch.xpu.get_rng_state()), "RNG states improperly synchronized on XPU."
    generator = torch.Generator()
    synchronize_rng_states(["generator"], generator=generator)
    assert are_the_same_tensors(generator.get_state()), "RNG states improperly synchronized in generator."

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

    print(state.process_index, result, type(dl))
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), "Wrong non-shuffled dataloader result."

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
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), "Wrong non-shuffled dataloader result."

    if state.process_index == 0:
        print("Non-shuffled dataloader passing.")

    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(dl, state.device, state.num_processes, state.process_index, put_on_device=True)
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length)), "Wrong shuffled dataloader result."

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
    assert result == list(range(length)), "Wrong shuffled dataloader result."

    if state.local_process_index == 0:
        print("Shuffled dataloader passing.")


def central_dl_preparation_check():
    state = AcceleratorState()
    length = 32 * state.num_processes

    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(
        dl, state.device, state.num_processes, state.process_index, put_on_device=True, dispatch_batches=True
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), "Wrong non-shuffled dataloader result."

    dl = DataLoader(range(length), batch_size=8)
    dl = prepare_data_loader(
        dl,
        state.device,
        state.num_processes,
        state.process_index,
        put_on_device=True,
        split_batches=True,
        dispatch_batches=True,
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result)
    assert torch.equal(result.cpu(), torch.arange(0, length).long()), "Wrong non-shuffled dataloader result."

    if state.process_index == 0:
        print("Non-shuffled central dataloader passing.")

    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(
        dl, state.device, state.num_processes, state.process_index, put_on_device=True, dispatch_batches=True
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length)), "Wrong shuffled dataloader result."

    dl = DataLoader(range(length), batch_size=8, shuffle=True)
    dl = prepare_data_loader(
        dl,
        state.device,
        state.num_processes,
        state.process_index,
        put_on_device=True,
        split_batches=True,
        dispatch_batches=True,
    )
    result = []
    for batch in dl:
        result.append(gather(batch))
    result = torch.cat(result).tolist()
    result.sort()
    assert result == list(range(length)), "Wrong shuffled dataloader result."

    if state.local_process_index == 0:
        print("Shuffled central dataloader passing.")


def mock_training(length, batch_size, generator):
    set_seed(42)
    generator.manual_seed(42)
    train_set = RegressionDataset(length=length, seed=42)
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
    assert are_the_same_tensors(old_model.a), "Did not obtain the same model on both processes."
    assert are_the_same_tensors(old_model.b), "Did not obtain the same model on both processes."

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
    assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on CPU or distributed training."
    assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on CPU or distributed training."

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
    assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on CPU or distributed training."
    assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on CPU or distributed training."

    accelerator.print("Training yielded the same results on one CPU or distributes setup with batch split.")

    if torch.cuda.is_available() or is_npu_available():
        # Mostly a test that FP16 doesn't crash as the operation inside the model is not converted to FP16
        print("FP16 training check.")
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16")
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
        assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on CPU or distributed training."
        assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on CPU or distributed training."

    if torch.cuda.is_available():
        # Mostly a test that model.forward will have autocast when running unwrap_model(model, keep_fp32_wrapper=True)
        print("Keep fp32 wrapper check.")
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16")

        model = torch.nn.Linear(2, 4)
        model = accelerator.prepare(model)
        model_with_fp32_wrapper = accelerator.unwrap_model(model, keep_fp32_wrapper=True)

        # Run forward with fp16 as input.
        # When the model is with mixed precision wrapper, no error will be raised.
        input_tensor = torch.Tensor([1, 2]).to(dtype=torch.float16, device=accelerator.device)
        output = model_with_fp32_wrapper(input_tensor)

    # BF16 support is only for CPU + TPU, and some GPU
    if is_bf16_available():
        # Mostly a test that BF16 doesn't crash as the operation inside the model is not converted to BF16
        print("BF16 training check.")
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="bf16")
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
        assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on CPU or distributed training."
        assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on CPU or distributed training."

    # IPEX support is only for CPU
    if is_ipex_available():
        print("ipex BF16 training check.")
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="bf16", cpu=True)
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
        assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on CPU or distributed training."
        assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on CPU or distributed training."

    # XPU support is only for XPU
    if is_xpu_available():
        print("xpu BF16 training check.")
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="bf16", cpu=False)
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
        assert torch.allclose(old_model.a, model.a), "Did not obtain the same model on XPU or distributed training."
        assert torch.allclose(old_model.b, model.b), "Did not obtain the same model on XPU or distributed training."


def test_split_between_processes_list():
    state = AcceleratorState()
    data = list(range(0, 2 * state.num_processes))
    with state.split_between_processes(data) as results:
        assert (
            len(results) == 2
        ), f"Each process did not have two items. Process index: {state.process_index}; Length: {len(results)}"

    data = list(range(0, (3 * state.num_processes) - 1))
    with state.split_between_processes(data, apply_padding=True) as results:
        if state.is_last_process:
            # Test that the last process gets the extra item(s)
            num_samples_per_device = math.ceil(len(data) / state.num_processes)
            assert (
                len(results) == num_samples_per_device
            ), f"Last process did not get the extra item(s). Process index: {state.process_index}; Length: {len(results)}"
    state.wait_for_everyone()


def test_split_between_processes_nested_dict():
    state = AcceleratorState()
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = ["a", "b", "c", "d", "e", "f", "g", "h"]
    c = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    if state.num_processes in (1, 2, 4):
        data = {"a": a, "b": b, "c": c}
        data_copy = deepcopy(data)
        with state.split_between_processes(data) as results:
            if state.process_index == 0:
                assert results["a"] == data_copy["a"][: 8 // state.num_processes]
            elif state.num_processes == 2:
                assert results["a"] == data_copy["a"][4:]
            elif state.process_index == 3:
                # We return a list each time
                assert results["a"] == data_copy["a"][-2:], f'Expected: {data_copy["a"][-2]}, Actual: {results["a"]}'
            if state.process_index == 0:
                assert results["b"] == data_copy["b"][: 8 // state.num_processes]
            elif state.num_processes == 2:
                assert results["b"] == data_copy["b"][4:]
            elif state.process_index == 3:
                assert results["b"] == data_copy["b"][-2:]
            if state.process_index == 0:
                assert torch.allclose(
                    results["c"], data_copy["c"][: 8 // state.num_processes]
                ), f"Did not obtain expected values on process 0, expected `{data['c'][:8 // state.num_processes]}`, received: {results['c']}"
            elif state.num_processes == 2:
                assert torch.allclose(
                    results["c"], data_copy["c"][4:]
                ), f"Did not obtain expected values on process 2, expected `{data['c'][4:]}`, received: {results['c']}"
            elif state.process_index == 3:
                assert torch.allclose(
                    results["c"], data_copy["c"][-2:]
                ), f"Did not obtain expected values on process 4, expected `{data['c'][-2:]}`, received: {results['c']}"

    state.wait_for_everyone()


def test_split_between_processes_tensor():
    state = AcceleratorState()
    if state.num_processes > 1:
        data = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]).to(state.device)
        with state.split_between_processes(data) as results:
            if state.process_index == 0:
                assert torch.allclose(results, torch.tensor([0, 1, 2, 3]).to(state.device))
            else:
                assert torch.allclose(results, torch.tensor([4, 5, 6, 7]).to(state.device))
    state.wait_for_everyone()


def test_trigger():
    accelerator = Accelerator()
    # should start with being false
    assert accelerator.check_trigger() is False

    # set a breakpoint on the main process
    if accelerator.is_main_process:
        accelerator.set_trigger()

    # check it's been activated across all processes
    # calls `all_reduce` and triggers a sync
    assert accelerator.check_trigger() is True

    # check it's been reset after the sync
    assert accelerator.check_trigger() is False


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.local_process_index == 0:
        print("**Initialization**")
    init_state_check()
    state.wait_for_everyone()

    if state.distributed_type == DistributedType.MULTI_GPU:
        num_processes_per_node = torch.cuda.device_count()
    else:
        num_processes_per_node = state.num_processes

    # We only run this test on non-multinode
    if num_processes_per_node == state.num_processes:
        if state.process_index == 0:
            print("\n**Test process execution**")
        process_execution_check()

        if state.process_index == 0:
            print("\n**Test split between processes as a list**")
        test_split_between_processes_list()

        if state.process_index == 0:
            print("\n**Test split between processes as a dict**")
        test_split_between_processes_nested_dict()

        if state.process_index == 0:
            print("\n**Test split between processes as a tensor**")
        test_split_between_processes_tensor()

    if state.local_process_index == 0:
        print("\n**Test random number generator synchronization**")
    rng_sync_check()

    if state.local_process_index == 0:
        print("\n**DataLoader integration test**")
    dl_preparation_check()
    if state.distributed_type != DistributedType.TPU:
        central_dl_preparation_check()

    # Trainings are not exactly the same in DeepSpeed and CPU mode
    if state.distributed_type == DistributedType.DEEPSPEED:
        return

    if state.local_process_index == 0:
        print("\n**Training integration test**")
    training_check()

    if state.local_process_index == 0:
        print("\n**Breakpoint trigger test**")
    test_trigger()


if __name__ == "__main__":
    main()
