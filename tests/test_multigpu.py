import inspect
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.utils import gather, set_seed, synchronize_rng_states
from testing_utils import are_the_same_tensors, execute_subprocess_async, require_multi_gpu
from training_utils import RegressionDataset, RegressionModel


class MultiGPUTester(unittest.TestCase):
    def setUp(self):
        self.test_file_path = inspect.getfile(self.__class__)

    @require_multi_gpu
    def test_multi_gpu(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        distributed_args = f"""
            -m torch.distributed.launch
            --nproc_per_node={torch.cuda.device_count()}
            {self.test_file_path}
        """.split()
        cmd = [sys.executable] + distributed_args
        execute_subprocess_async(cmd, env=os.environ.copy())


def init_state_check():
    # Test we can instantiate this twice in a row.
    state = AcceleratorState()
    if state.local_process_index == 0:
        print("Testing, testing. 1, 2, 3.")
    print(state)


def rng_sync_check():
    state = AcceleratorState()
    synchronize_rng_states()
    assert are_the_same_tensors(torch.get_rng_state())
    assert are_the_same_tensors(torch.cuda.get_rng_state())
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


def mock_training(length, batch_size):
    set_seed(42)
    train_set = RegressionDataset(length=length)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            loss.backward()
            optimizer.step()
    return train_set, model


def training_check():
    state = AcceleratorState()
    batch_size = 8
    length = batch_size * 4 * state.num_processes

    train_set, old_model = mock_training(length, batch_size * state.num_processes)
    assert are_the_same_tensors(old_model.a)
    assert are_the_same_tensors(old_model.b)

    accelerator = Accelerator()
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            loss.backward()
            optimizer.step()

    model = model.module.cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)

    accelerator.print("Training yielded the same results on one CPU or 2 GPUs with no batch split.")

    accelerator = Accelerator(split_batches=True)
    train_dl = DataLoader(train_set, batch_size=batch_size * state.num_processes, shuffle=True)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            loss.backward()
            optimizer.step()

    model = model.module.cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)

    accelerator.print("Training yielded the same results on one CPU or 2 GPUs with batch split.")

    # Mostly a test that FP16 doesn't crash as the operation inside the model is not converted to FP16
    accelerator = Accelerator(fp16=True)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch["x"])
            loss = torch.nn.functional.mse_loss(output, batch["y"])
            accelerator.backward(loss)
            optimizer.step()

    model = model.module.cpu()
    assert torch.allclose(old_model.a, model.a)
    assert torch.allclose(old_model.b, model.b)


if __name__ == "__main__":
    state = AcceleratorState()
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
