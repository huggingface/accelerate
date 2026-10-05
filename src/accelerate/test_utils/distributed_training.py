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

"""Launch training workers and read their results; scenarios own the assertions."""

import json
import os
import sys

from .testing import (
    execute_subprocess_async,
    get_launch_command,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
)


def run_training(
    output,
    *,
    batch_size,
    config_file=None,
    reference=False,
    mixed_precision="no",
    gradient_accumulation_steps=1,
):
    """Run a training worker and return its measurements; the caller checks the results."""
    # Require an explicit setup instead of using the machine's default launch configuration.
    if not reference and config_file is None:
        raise ValueError("Distributed training requires an explicit launch configuration.")
    if reference and config_file is not None:
        raise ValueError("The plain-PyTorch reference does not use a launch configuration.")

    command = [sys.executable]
    if not reference:
        command = get_launch_command(
            config_file=config_file,
            mixed_precision=mixed_precision,
            main_process_port=get_torch_dist_unique_port(),
        )
    command += [
        str(path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_causal_lm.py")),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--mixed-precision",
        mixed_precision,
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
    ]
    if reference:
        command.append("--reference")

    result = execute_subprocess_async(command, env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))
