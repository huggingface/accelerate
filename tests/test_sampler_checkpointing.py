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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed


@pytest.mark.parametrize("seedable", [False, True])
@pytest.mark.parametrize("generator_kind", ["global", "shared", "separate"])
@pytest.mark.parametrize("epoch,completed", [(0, 3), (1, 3), (0, 8)])
def test_checkpoint_restores_sampler_and_training_rng(tmp_path, seedable, generator_kind, epoch, completed):
    env = {**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1"}
    for phase in ("save", "resume"):
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                phase,
                str(tmp_path),
                str(int(seedable)),
                generator_kind,
                str(epoch),
                str(completed),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("explicit_generator", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
def test_checkpointable_sampler_preserves_uninterrupted_rng(explicit_generator, replacement):
    from accelerate.data_loader import _CheckpointableRandomSampler

    results = []
    for wrapped in (False, True):
        set_seed(42)
        generator = torch.Generator().manual_seed(7) if explicit_generator else None
        sampler = RandomSampler(range(15), replacement=replacement, num_samples=21, generator=generator)
        if wrapped:
            sampler = _CheckpointableRandomSampler(sampler)
        results.append(([list(sampler) for _ in range(3)], torch.rand(5).tolist()))
    assert results[0] == results[1]


@pytest.mark.parametrize("explicit_generator", [False, True])
def test_checkpointable_sampler_preserves_overlapping_iterators(explicit_generator):
    from accelerate.data_loader import _CheckpointableRandomSampler

    results = []
    for wrapped in (False, True):
        set_seed(42)
        generator = torch.Generator().manual_seed(7) if explicit_generator else None
        sampler = RandomSampler(range(15), generator=generator)
        if wrapped:
            sampler = _CheckpointableRandomSampler(sampler)
        first = iter(sampler)
        prefix = next(first)
        second = list(sampler)
        results.append((prefix, second, list(first), torch.rand(5).tolist()))
    assert results[0] == results[1]


@pytest.mark.parametrize("seedable", [False, True])
def test_checkpoint_during_resumed_epoch(tmp_path, seedable):
    env = {**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1"}
    for phase in ("save", "recheckpoint", "resume_again", "legacy"):
        result = subprocess.run(
            [sys.executable, __file__, phase, str(tmp_path), str(int(seedable)), "global", "1", "3"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def worker(phase, directory, seedable, generator_kind, epoch, completed):
    directory = Path(directory)
    epoch, completed = int(epoch), int(completed)
    set_seed(42)
    accelerator = Accelerator(
        cpu=True, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=bool(int(seedable)))
    )
    dataset = list(range(32))
    if generator_kind == "separate":
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(7))
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    else:
        generator = torch.Generator().manual_seed(7) if generator_kind == "shared" else None
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
    model = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    checkpoint = directory / ("checkpoint_again" if phase == "resume_again" else "checkpoint")
    if phase == "legacy":
        for sidecar in checkpoint.glob("dl_sampler_state_*.bin"):
            sidecar.unlink()
        accelerator.load_state(checkpoint)
        return
    if phase == "resume_again":
        completed += 1
    if phase != "save":
        accelerator.load_state(checkpoint)
    records = []
    start = epoch + (completed == len(loader)) if phase != "save" else 0
    for current_epoch in range(start, 3):
        active = (
            accelerator.skip_first_batches(loader, completed) if phase != "save" and current_epoch == epoch else loader
        )
        for step, batch in enumerate(active):
            inputs = batch.float().reshape(-1, 1) / 32
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), inputs.square())
            accelerator.backward(loss)
            optimizer.step()
            weights = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).tolist()
            records.append([batch.tolist(), torch.rand(5).tolist(), loss.item(), weights])
            if phase == "save" and current_epoch == epoch and step + 1 == completed:
                accelerator.save_state(checkpoint)
            if phase == "recheckpoint" and current_epoch == epoch and step == 0:
                accelerator.save_state(directory / "checkpoint_again")
    expected_path = directory / f"expected_{accelerator.process_index}.json"
    if phase == "save":
        expected_path.write_text(json.dumps(records[epoch * len(loader) + completed :]))
    else:
        expected = json.loads(expected_path.read_text())
        if phase == "resume_again":
            expected = expected[1:]
        assert records == expected, "Samples, training RNG, loss or weights diverged after resume"


if __name__ == "__main__":
    worker(*sys.argv[1:])
