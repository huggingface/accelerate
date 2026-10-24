# Copyright 2026 Daniel Wahnich. All rights reserved.
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

"""Exercise source ownership at the actual FSDP2 assignment boundary on two GPUs."""

import json
from unittest.mock import patch

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard

from accelerate import Accelerator, FullyShardedDataParallelPlugin, ParallelismConfig
from accelerate.utils.fsdp_utils import fsdp2_load_full_state_dict


def main():
    accelerator = Accelerator(
        fsdp_plugin=FullyShardedDataParallelPlugin(fsdp_version=2, cpu_ram_efficient_loading=True),
        parallelism_config=ParallelismConfig(dp_shard_size=2),
    )
    assert accelerator.num_processes == 2
    rank = accelerator.process_index
    mesh = init_device_mesh("cuda", (2,))
    cases = 0
    for rows in (1, 9):
        for cpu_offload in (False, True):
            for nonmain_source in ("empty", "meta", "different"):
                for consume in (False, True):
                    source = torch.nn.Sequential(torch.nn.Linear(8, rows))
                    with torch.no_grad():
                        for index, value in enumerate(source.parameters()):
                            value.copy_(torch.arange(value.numel()).reshape(value.shape) + index * 10 + rank * 100)
                    full = source.state_dict()
                    if rank == 1 and nonmain_source == "empty":
                        full.clear()
                    elif rank == 1 and nonmain_source == "meta":
                        full = {name: value.to("meta") for name, value in full.items()}
                    before = {name: value.clone() for name, value in full.items()}
                    model = torch.nn.Sequential(torch.nn.Linear(8, rows, device="meta"))
                    offload = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
                    fully_shard(model, mesh=mesh, offload_policy=offload)
                    native_load = model.load_state_dict
                    assignment_calls = []

                    def checked_load(*args, **kwargs):
                        assignment_calls.append(True)
                        assert bool(full) == (bool(before) and not consume), (
                            "Source ownership violated before assignment"
                        )
                        return native_load(*args, **kwargs)

                    # Intercept only to inspect the lifetime; the real native assignment still executes.
                    with patch.object(model, "load_state_dict", side_effect=checked_load):
                        options = {"consume_full_state_dict": True} if consume else {}
                        fsdp2_load_full_state_dict(accelerator, model, full, cpu_offload=cpu_offload, **options)
                    assert len(assignment_calls) == 1
                    if consume:
                        assert not full
                    else:
                        assert full.keys() == before.keys()
                        for name, value in before.items():
                            assert full[name].device == value.device
                            if value.device.type != "meta":
                                torch.testing.assert_close(full[name], value, rtol=0, atol=0)
                    for index, value in enumerate(model.state_dict().values()):
                        expected = torch.arange(value.numel()).reshape(value.shape) + index * 10
                        start = rank * ((rows + 1) // 2)
                        expected = expected[start : start + (rows + 1) // 2].float()
                        local = value.to_local()
                        assert local.device.type == ("cpu" if cpu_offload else "cuda")
                        if cpu_offload and local.numel():
                            assert local.is_pinned()
                        torch.testing.assert_close(local.cpu(), expected, rtol=0, atol=0)
                    cases += 1
    accelerator.wait_for_everyone()
    print(json.dumps({"rank": rank, "cases": cases, "status": "passed"}), flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
