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
import itertools
import os
import socket
from collections import namedtuple
from datetime import timedelta

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import InitProcessGroupKwargs, broadcast


Payload = namedtuple("Payload", ["values"])


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _accelerator(rank, port, dispatch=False):
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
    return Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=dispatch, split_batches=True),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=30))],
    )


def _view(layout, dtype, shape=(2, 3)):
    offset = 2
    if layout == "transpose":
        stride = (1, shape[0])
    elif layout == "strided":
        stride = (shape[1] * 2 + 1, 2)
    elif layout == "channels_last":
        stride = (12, 1, 6, 3)
    else:
        stride = tuple(
            int(torch.tensor(shape[axis + 1 :]).prod()) if axis + 1 < len(shape) else 1 for axis in range(len(shape))
        )
    positions = [
        offset + sum(index * step for index, step in zip(indices, stride))
        for indices in itertools.product(*(range(size) for size in shape))
    ]
    size = max(positions, default=offset) + 3
    backing = torch.full((size,), -7, dtype=dtype)
    tensor = backing.as_strided(shape, stride, offset)
    return tensor, backing, positions


def _broadcast_worker(rank, port, dtype_name, views):
    _accelerator(rank, port)
    failures = []
    dtype = getattr(torch, dtype_name)
    layouts = [
        ("contiguous", "contiguous", (2, 3)),
        ("contiguous", "contiguous", ()),
        ("strided", "contiguous", (0, 3)),
    ]
    if views:
        layouts = [
            ("transpose", "contiguous", (2, 3)),
            ("contiguous", "transpose", (2, 3)),
            ("strided", "contiguous", (2, 3)),
            ("contiguous", "strided", (2, 3)),
            ("transpose", "strided", (2, 3)),
            ("strided", "transpose", (2, 3)),
            ("channels_last", "contiguous", (2, 3, 2, 2)),
            ("contiguous", "channels_last", (2, 3, 2, 2)),
        ]
    try:
        for source in (0, 1):
            for source_layout, destination_layout, shape in layouts:
                layout = source_layout if rank == source else destination_layout
                tensor, backing, positions = _view(layout, dtype, shape)
                expected = torch.arange(tensor.numel()).reshape(shape).to(dtype)
                tensor.copy_(expected if rank == source else torch.full_like(tensor, -3))
                expected_backing = backing.clone()
                if positions:
                    expected_backing[positions] = expected.flatten()
                original_stride = tensor.stride()
                original_pointer = tensor.data_ptr()
                # The public recursive operation must preserve both the container and
                # the original leaf, including its aliases into a larger allocation.
                result = broadcast({"nested": [Payload(tensor)]}, from_process=source)
                try:
                    assert isinstance(result["nested"][0], Payload)
                    assert result["nested"][0].values is tensor
                    assert tensor.stride() == original_stride
                    assert tensor.data_ptr() == original_pointer
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
                except AssertionError as error:
                    failures.append(f"source={source}, rank={rank}, {source_layout}->{destination_layout}: {error}")
        assert not failures, "\n".join(failures)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64", "int64"])
@pytest.mark.parametrize("views", [False, True], ids=["contiguous-controls", "noncontiguous-views"])
def test_broadcast_preserves_logical_values_and_aliases(dtype_name, views):
    torch.multiprocessing.spawn(_broadcast_worker, args=(_free_port(), dtype_name, views), nprocs=2)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64", "int64"])
def test_single_process_broadcast_keeps_views(dtype_name):
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    Accelerator(cpu=True)
    try:
        for layout in ("contiguous", "transpose", "strided"):
            tensor, backing, _ = _view(layout, getattr(torch, dtype_name))
            previous = backing.clone()
            assert broadcast(tensor) is tensor
            torch.testing.assert_close(backing, previous, rtol=0, atol=0)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()


def _source_graph_worker(rank, port, source, dtype_name, leaf_kind):
    _accelerator(rank, port)
    try:
        dtype = getattr(torch, dtype_name)
        values = (torch.arange(12, dtype=dtype) + rank * 20).reshape(3, 4)
        backing = values.requires_grad_() if leaf_kind == "view" else torch.nn.Parameter(values)
        tensor = backing.t()
        previous_loss = tensor.square().sum()
        previous_version = backing._version
        expected = (torch.arange(12, dtype=dtype) + source * 20).reshape(3, 4)
        assert broadcast(tensor, from_process=source) is tensor
        torch.testing.assert_close(backing, expected, rtol=0, atol=0)
        if rank == source:
            # Sending a view must not invalidate an already constructed source graph.
            assert backing._version == previous_version
            previous_loss.backward()
        else:
            # Receiving follows ordinary no_grad copy_ state-update semantics. For
            # a leaf that itself views a non-grad base, recreate forward views after
            # that update; a Parameter's existing view retains its local gradient.
            forward_view = backing.t() if leaf_kind == "view" else tensor
            forward_view.square().sum().backward()
        torch.testing.assert_close(backing.grad, 2 * expected, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("leaf_kind", ["view", "parameter"])
def test_broadcast_source_preserves_saved_forward_graph(source, dtype_name, leaf_kind):
    torch.multiprocessing.spawn(_source_graph_worker, args=(_free_port(), source, dtype_name, leaf_kind), nprocs=2)


class _Collate:
    def __init__(self, layout):
        self.layout = layout

    def __call__(self, examples):
        values, labels, indices = default_collate(examples)
        if self.layout == "transpose":
            values = values.transpose(1, 2).contiguous().transpose(1, 2)
        elif self.layout == "channels_last":
            values = values.contiguous(memory_format=torch.channels_last)
        return values, labels, indices


def _train_worker(rank, port, layout, optimizer_name, dtype_name):
    accelerator = _accelerator(rank, port, dispatch=True)
    try:
        dtype = getattr(torch, dtype_name)
        features = 12 if layout == "channels_last" else 6
        shape = (3, 2, 2) if layout == "channels_last" else (3, 2)
        values = torch.arange(12 * features, dtype=dtype).reshape(12, *shape) / 100
        labels = values.flatten(1)[:, :2] * 0.2 + 0.1
        torch.manual_seed(87)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(features, 2)).to(dtype)
        reference = copy.deepcopy(model)
        optimizer_class = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
        optimizer_kwargs = {"momentum": 0.9} if optimizer_name == "sgd" else {}
        optimizer = optimizer_class(model.parameters(), lr=0.03, **optimizer_kwargs)
        expected_optimizer = optimizer_class(reference.parameters(), lr=0.03, **optimizer_kwargs)
        loader = DataLoader(
            TensorDataset(values.clone(), labels.clone(), torch.arange(12)),
            batch_size=4,
            collate_fn=_Collate(layout),
        )
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        assert isinstance(loader, DataLoaderDispatcher)
        gathered_indices = []
        for step, (inputs, targets, indices) in enumerate(loader):
            optimizer.zero_grad()
            accelerator.backward((model(inputs) - targets).square().mean())
            expected_optimizer.zero_grad()
            selection = slice(step * 4, (step + 1) * 4)
            expected_loss = (reference(values[selection]) - labels[selection]).square().mean()
            expected_loss.backward()
            for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            optimizer.step()
            expected_optimizer.step()
            for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=1e-5, atol=1e-7)
            gathered_indices.extend(accelerator.gather_for_metrics(indices).tolist())
        assert step == 2
        assert gathered_indices == list(range(12))
        for actual in accelerator.unwrap_model(model).parameters():
            copies = [torch.empty_like(actual), torch.empty_like(actual)]
            torch.distributed.all_gather(copies, actual)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "channels_last"])
@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_dispatcher_broadcast_matches_global_batch_training(layout, optimizer_name, dtype_name):
    torch.multiprocessing.spawn(_train_worker, args=(_free_port(), layout, optimizer_name, dtype_name), nprocs=2)
