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

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import DataLoaderShard
from accelerate.test_utils.testing import require_torchdata_stateful_dataloader


def _stateful_partition_worker(rank, world_size, init_file, result_file):
    """Run one rank with intentionally different pre-prepare RNG state."""
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world_size)
    torch.manual_seed(1234 + rank)

    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True),
    )
    generator = torch.Generator().manual_seed(999)
    generator_state = generator.get_state()
    dataloader = accelerator.prepare(
        DataLoader(
            TensorDataset(torch.arange(96)),
            batch_size=4,
            shuffle=True,
            num_workers=2,
            generator=generator,
        )
    )
    torch.testing.assert_close(generator.get_state(), generator_state)
    assert dataloader.dl_state_dict is None
    values = [value for batch in dataloader for value in batch[0].tolist()]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, values)
    if rank == 0:
        union = [value for partition in gathered for value in partition]
        Path(result_file).write_text(
            json.dumps({"unique": len(set(union)), "duplicates": len(union) - len(set(union))}),
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()


@require_torchdata_stateful_dataloader
def test_stateful_adapter_defers_initial_sampler_snapshot():
    generator = torch.Generator().manual_seed(123)
    initial_state = generator.get_state()
    dataloader = DataLoaderShard(
        list(range(32)),
        batch_size=4,
        num_workers=2,
        generator=generator,
        use_stateful_dataloader=True,
    )

    assert dataloader.dl_state_dict is None
    torch.testing.assert_close(generator.get_state(), initial_state)
    list(dataloader)
    assert dataloader.dl_state_dict is not None
    assert dataloader.dl_state_dict["_iterator_finished"] is True


@require_torchdata_stateful_dataloader
def test_stateful_prepare_preserves_cross_rank_shuffle_partition(tmp_path):
    init_file = tmp_path / "process-group"
    result_file = tmp_path / "result.json"
    mp.spawn(
        _stateful_partition_worker,
        args=(2, str(init_file), str(result_file)),
        nprocs=2,
        join=True,
    )
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result == {"unique": 96, "duplicates": 0}
