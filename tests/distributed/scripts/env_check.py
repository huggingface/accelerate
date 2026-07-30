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
"""Dump the per-rank distributed environment seen by an ``Accelerator`` to JSON.

Each rank instantiates an ``Accelerator``, builds a small record of what it perceives
(distributed_type, rank/local_rank, device, mixed_precision, plus a few raw env vars),
and the main process writes all records sorted by ``process_index`` to ``--output_file``.

Every value is coerced to a JSON-serializable scalar; enum values are ``str()``-ified
(e.g., ``"DistributedType.MULTI_GPU"``).
"""

import argparse
import json
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import gather_object


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_file", type=Path, required=True)
    return p.parse_args()


def build_env_record(accelerator: Accelerator) -> dict:
    return {
        "distributed_type": str(accelerator.distributed_type),
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "local_process_index": accelerator.local_process_index,
        "device": str(accelerator.device),
        "mixed_precision": str(accelerator.mixed_precision),
        # Raw env-vars set by the launcher, for cross-launcher parity comparison.
        "env_WORLD_SIZE": int(os.environ.get("WORLD_SIZE", "1")),
        "env_RANK": int(os.environ.get("RANK", "0")),
        # local_rank can legitimately be -1 before the framework consumes it.
        "env_LOCAL_RANK": int(os.environ.get("LOCAL_RANK", "0")),
        "env_MASTER_ADDR": os.environ.get("MASTER_ADDR", ""),
    }


def main():
    args = parse_args()
    accelerator = Accelerator()

    record = build_env_record(accelerator)
    all_records = gather_object([record])

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_records.sort(key=lambda r: r["process_index"])
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(all_records, indent=2))


if __name__ == "__main__":
    main()
