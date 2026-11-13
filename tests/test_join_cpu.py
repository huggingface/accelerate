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
import os
import socket
import warnings
from datetime import timedelta

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _join_cpu_worker(rank, port, size, override, fail):
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
        dataloader_config=DataLoaderConfiguration(even_batches=override is not None),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=30))],
    )
    try:
        torch.manual_seed(123)
        model = torch.nn.Linear(1, 1)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        values = torch.arange(1, size + 1, dtype=torch.float32).reshape(-1, 1) / 10
        targets = values * 0.2 + 0.3
        loader = DataLoader(TensorDataset(values, targets, torch.arange(size)), batch_size=1)
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        original_even_batches = loader.batch_sampler.even_batches
        seen = []
        with warnings.catch_warnings(record=True) as caught:
            try:
                with accelerator.join_uneven_inputs([model], even_batches=override):
                    # Fail before entering uneven collectives on versions that only provide a no-op here.
                    assert not any("will have no effect" in str(item.message) for item in caught)
                    if override is not None:
                        assert loader.batch_sampler.even_batches == override
                    if fail:
                        raise RuntimeError("intentional failure")
                    for inputs, labels, indices in loader:
                        optimizer.zero_grad()
                        accelerator.backward((model(inputs) - labels).square().mean())
                        optimizer.step()
                        seen.extend(indices.tolist())
            except RuntimeError as error:
                if not fail or str(error) != "intentional failure":
                    raise
            else:
                assert not fail
        assert loader.batch_sampler.even_batches == original_even_batches

        if not fail:
            seen_by_rank = [None, None]
            torch.distributed.all_gather_object(seen_by_rank, seen)
            assert sorted(seen_by_rank[0] + seen_by_rank[1]) == list(range(size))
            assert [len(indices) for indices in seen_by_rank] == [(size + 1) // 2, size // 2]

            # Join uses the initial world size for gradient averaging, including the final unpaired sample.
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
            for offset in range(0, size, 2):
                reference_optimizer.zero_grad()
                loss = (reference(values[offset : offset + 2]) - targets[offset : offset + 2]).square().sum() / 2
                loss.backward()
                reference_optimizer.step()
            for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                copies = [torch.empty_like(actual), torch.empty_like(actual)]
                torch.distributed.all_gather(copies, actual)
                torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize("override", [None, False])
def test_join_uneven_cpu_ddp_matches_training_reference(size, override):
    torch.multiprocessing.spawn(_join_cpu_worker, args=(_free_port(), size, override, False), nprocs=2)


@pytest.mark.parametrize("override", [None, False])
def test_join_cpu_ddp_restores_sampler_after_exception(override):
    torch.multiprocessing.spawn(_join_cpu_worker, args=(_free_port(), 3, override, True), nprocs=2)


def test_join_single_process_keeps_noop_behavior():
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        accelerator = Accelerator(cpu=True)
        with warnings.catch_warnings(record=True) as caught:
            with accelerator.join_uneven_inputs([torch.nn.Linear(1, 1)], even_batches=False):
                pass
        assert not caught
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
