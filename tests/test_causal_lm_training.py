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
import sys

import torch

from accelerate.commands.config.config_args import ClusterConfig
from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    get_launch_command,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
    require_cuda,
    require_multi_device,
    require_transformers,
)
from accelerate.utils import ComputeEnvironment, DistributedType, patch_environment


@require_cuda
@require_multi_device
@require_transformers
class CausalLMTrainingTester(TempDirTestCase):
    def test_single_process_matches_ddp(self):
        """A real causal-LM loop preserves losses and updates at equal effective batch size."""
        script = path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_causal_lm.py")
        single_path, ddp_path = self.tmpdir / "single.json", self.tmpdir / "ddp.json"
        config_path = self.tmpdir / "ddp_config.json"
        # Do not inherit a developer's default launcher configuration.
        ClusterConfig(
            compute_environment=ComputeEnvironment.LOCAL_MACHINE,
            distributed_type=DistributedType.MULTI_GPU,
            mixed_precision="no",
            debug=False,
            use_cpu=False,
            num_processes=2,
        ).to_json_file(config_path)
        ddp_command = get_launch_command(config_file=str(config_path), main_process_port=get_torch_dist_unique_port())

        with patch_environment(omp_num_threads=1, cublas_workspace_config=":4096:8", hf_hub_offline="1"):
            execute_subprocess_async([sys.executable, script, "--output", str(single_path)], timeout=90)
            execute_subprocess_async(ddp_command + [script, "--output", str(ddp_path)], timeout=90)

        single, ddp = json.loads(single_path.read_text()), json.loads(ddp_path.read_text())
        self.assertEqual(single["world_size"], 1)
        self.assertEqual(ddp["world_size"], 2)
        self.assertEqual(len(single["losses"]), 10)
        self.assertEqual(len(ddp["losses"]), 10)
        self.assertTrue(single["parameters"])
        # Reduction order can differ between one process and DDP. Compare the
        # complete trajectory and final updates, not just successful execution.
        for field, atol in (("losses", 1e-4), ("parameters", 1e-5)):
            with self.subTest(field=field):
                torch.testing.assert_close(
                    torch.tensor(single[field], dtype=torch.float64),
                    torch.tensor(ddp[field], dtype=torch.float64),
                    rtol=0,
                    atol=atol,
                )
