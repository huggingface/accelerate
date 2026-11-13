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

import copy
import gc
import os
import socket
import weakref
from collections import namedtuple
from datetime import timedelta

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import InitProcessGroupKwargs, slice_tensors


Sample = namedtuple("Sample", ["features", "index"])


def _leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    else:
        for item in value:
            yield from _leaves(item)


class _Samples(Dataset):
    def __init__(self, structure, dtype):
        self.structure = structure
        self.dtype = dtype

    def __len__(self):
        return 48

    def __getitem__(self, index):
        features = torch.arange(32, dtype=self.dtype) + index
        index_tensor = torch.tensor(index)
        if self.structure == "mapping":
            return {"features": features, "metadata": [index_tensor, index_tensor % 2 == 0]}
        if self.structure == "namedtuple":
            return Sample(features, index_tensor)
        return features


class _TrackFirstBatch:
    def __init__(self):
        self.references = None

    def __call__(self, batch, tensor_slice, **kwargs):
        if self.references is None:
            self.references = [weakref.ref(tensor) for tensor in _leaves(batch)]
        return slice_tensors(batch, tensor_slice, **kwargs)


@pytest.fixture
def cpu_accelerator():
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    accelerator = Accelerator(cpu=True)
    yield accelerator
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.mark.parametrize("structure", ["tensor", "mapping", "namedtuple"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("split_batches", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_dispatcher_releases_first_full_batch(cpu_accelerator, structure, dtype, split_batches, drop_last):
    tracker = _TrackFirstBatch()
    loader = DataLoaderDispatcher(
        _Samples(structure, dtype),
        batch_size=16,
        split_batches=split_batches,
        drop_last=drop_last,
        _drop_last=drop_last,
        slice_fn=tracker,
    )
    iterator = iter(loader)
    first = next(iterator)
    del first
    second = next(iterator)
    del second
    gc.collect()
    try:
        # Only weak references are retained by the observer. The padding cache must not
        # keep the original full-batch tensors alive after their consumer releases them.
        assert all(reference() is None for reference in tracker.references)
    finally:
        list(iterator)
    assert not cpu_accelerator.gradient_state.in_dataloader


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _train_worker(rank, port, split_batches, optimizer_name, dtype_name):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        ACCELERATE_USE_CPU="true",
        ACCELERATE_MIXED_PRECISION="no",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=split_batches),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=30))],
    )
    try:
        dtype = getattr(torch, dtype_name)
        torch.manual_seed(314)
        model = torch.nn.Linear(4, 2).to(dtype)
        reference = copy.deepcopy(model)
        optimizer_class = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
        optimizer_kwargs = {"momentum": 0.9} if optimizer_name == "sgd" else {}
        optimizer = optimizer_class(model.parameters(), lr=0.03, **optimizer_kwargs)
        expected_optimizer = optimizer_class(reference.parameters(), lr=0.03, **optimizer_kwargs)
        values = torch.arange(36, dtype=dtype).reshape(9, 4) / 40
        targets = torch.stack((values[:, 0] * 0.2, values[:, 1] * -0.1), dim=-1)
        loader = DataLoader(
            TensorDataset(values.clone(), targets.clone(), torch.arange(9)),
            batch_size=4 if split_batches else 2,
        )
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        assert isinstance(loader, DataLoaderDispatcher)
        gathered_indices = []
        expected_batches = [list(range(4)), list(range(4, 8)), [8, 0]]
        for step, (inputs, labels, indices) in enumerate(loader):
            # A common in-place input normalization must retain the same training result.
            inputs.mul_(0.5).add_(0.2)
            optimizer.zero_grad()
            accelerator.backward((model(inputs) - labels).square().mean())
            expected_optimizer.zero_grad()
            selection = expected_batches[step]
            expected_inputs = values[selection] * 0.5 + 0.2
            expected_loss = (reference(expected_inputs) - targets[selection]).square().mean()
            expected_loss.backward()
            for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            optimizer.step()
            expected_optimizer.step()
            for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=1e-5, atol=1e-7)
            gathered_indices.extend(accelerator.gather_for_metrics(indices).tolist())
        assert gathered_indices == list(range(9))
        assert step == 2
        assert not accelerator.gradient_state.in_dataloader
        for actual in accelerator.unwrap_model(model).parameters():
            copies = [torch.empty_like(actual), torch.empty_like(actual)]
            torch.distributed.all_gather(copies, actual)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("split_batches", [False, True])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_dispatcher_prefix_copy_preserves_ddp_training(split_batches, optimizer_name, dtype_name):
    torch.multiprocessing.spawn(
        _train_worker, args=(_free_port(), split_batches, optimizer_name, dtype_name), nprocs=2
    )
