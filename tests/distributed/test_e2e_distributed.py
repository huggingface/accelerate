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
"""End-to-end distributed tests for ``accelerate``.

Structure per backend (three pieces):

    * ``{Backend}CommandsMixin`` — builds the ``accelerate launch`` and ``torchrun`` commands.
    * ``TestDistributed{Backend}`` — backend-specific, non-slow tests (env parity).
    * ``TestDistributed{Backend}Common`` — shared ``check_*`` scenarios, ``@slow``.
"""

import json
import math
from pathlib import Path

from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    get_launch_command,
    get_torch_dist_unique_port,
    require_multi_device,
    require_non_torch_xla,
    require_transformers,
    slow,
)


HERE = Path(__file__).parent
CONFIGS_DIR = HERE / "accelerate_configs"
SCRIPTS_DIR = HERE / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train.py"
ENV_CHECK_SCRIPT = SCRIPTS_DIR / "env_check.py"


class DDPCommandsMixin:
    config_file = CONFIGS_DIR / "ddp.yaml"
    num_processes = 2

    def get_accelerate_cmd(self, script, *args):
        cmd = get_launch_command(config_file=str(self.config_file))
        cmd.extend([str(script), *args])
        return cmd

    def get_torchrun_cmd(self, script, *args):
        return [
            "torchrun",
            f"--nproc_per_node={self.num_processes}",
            "--nnodes=1",
            f"--master_port={get_torch_dist_unique_port()}",
            str(script),
            *args,
        ]


class DistributedCommon:
    """Shared scenarios. Subclasses provide ``get_accelerate_cmd`` via a Mixin."""

    def check_smoke(self):
        output = self.tmpdir / "smoke.json"
        cmd = self.get_accelerate_cmd(
            TRAIN_SCRIPT,
            f"--output_file={output}",
            "--max_steps=4",
        )
        execute_subprocess_async(cmd, timeout=180)

        assert output.exists(), f"train.py did not write {output}"
        payload = json.loads(output.read_text())
        assert payload["num_steps"] == 4, payload
        assert payload["num_processes"] == self.num_processes, payload
        assert len(payload["loss_history"]) == 4, payload
        assert math.isfinite(payload["final_loss"]), payload
        assert all(math.isfinite(x) for x in payload["loss_history"]), payload


@require_non_torch_xla
@require_multi_device
@require_transformers
class TestDistributedDDP(DDPCommandsMixin, TempDirTestCase):
    """Non-slow DDP tests: env parity between ``accelerate launch`` and ``torchrun``."""

    def test_env_parity_accelerate_vs_torchrun(self):
        acc_out = self.tmpdir / "env_accelerate.json"
        tr_out = self.tmpdir / "env_torchrun.json"

        acc_cmd = self.get_accelerate_cmd(ENV_CHECK_SCRIPT, f"--output_file={acc_out}")
        execute_subprocess_async(acc_cmd, timeout=60)

        tr_cmd = self.get_torchrun_cmd(ENV_CHECK_SCRIPT, f"--output_file={tr_out}")
        execute_subprocess_async(tr_cmd, timeout=60)

        acc = json.loads(acc_out.read_text())
        tr = json.loads(tr_out.read_text())

        assert len(acc) == self.num_processes, acc
        assert len(tr) == self.num_processes, tr

        for a, t in zip(acc, tr):
            assert a["num_processes"] == t["num_processes"] == self.num_processes
            assert a["process_index"] == t["process_index"]
            assert a["local_process_index"] == t["local_process_index"]
            assert a["distributed_type"] == t["distributed_type"]
            assert a["device"] == t["device"]
            assert a["mixed_precision"] == t["mixed_precision"]
            assert a["env_WORLD_SIZE"] == t["env_WORLD_SIZE"]
            assert a["env_RANK"] == t["env_RANK"]
            # LOCAL_RANK may be -1 until the framework consumes it — accept either.
            assert a["env_LOCAL_RANK"] in (t["env_LOCAL_RANK"], -1)


@require_non_torch_xla
@require_multi_device
@require_transformers
@slow
class TestDistributedDDPCommon(DDPCommandsMixin, DistributedCommon, TempDirTestCase):
    """Slow DDP scenarios that launch ``train.py`` via ``accelerate launch`` and assert on its JSON output."""

    def test_smoke(self):
        self.check_smoke()
