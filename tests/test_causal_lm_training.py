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

import pytest
import torch

from accelerate.commands.config.config_args import ClusterConfig
from accelerate.test_utils.testing import (
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
@pytest.mark.parametrize(
    "mixed_precision, gradient_accumulation_steps, loss_atol, parameter_atol",
    [
        pytest.param("no", 1, 1e-4, 1e-5, id="fp32_ddp"),
        pytest.param("no", 2, 1e-4, 1e-5, id="fp32_accumulation"),
        pytest.param("bf16", 2, 1e-2, 1e-3, id="bf16_accumulation"),
        pytest.param("fp16", 2, 1e-3, 1e-4, id="fp16_accumulation"),
    ],
)
def test_ddp_training_matches_reference(
    tmp_path, mixed_precision, gradient_accumulation_steps, loss_atol, parameter_atol
):
    """DDP, accumulation and AMP match a same-precision plain-PyTorch reference."""
    pytest.importorskip("transformers.models.gemma4", reason="Requires Transformers with Gemma 4 support")
    if mixed_precision == "bf16":
        for device in range(2):
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    pytest.skip("Both CUDA devices must support BF16")

    script_path = path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_causal_lm.py")
    reference_results_path, ddp_results_path = tmp_path / "reference.json", tmp_path / "ddp.json"
    config_path = tmp_path / "ddp_config.json"
    # Do not inherit a developer's default launcher configuration.
    ClusterConfig(
        compute_environment=ComputeEnvironment.LOCAL_MACHINE,
        distributed_type=DistributedType.MULTI_GPU,
        mixed_precision=mixed_precision,
        debug=False,
        use_cpu=False,
        num_processes=2,
    ).to_json_file(config_path)
    launch_command = get_launch_command(config_file=str(config_path), main_process_port=get_torch_dist_unique_port())
    training_args = [
        "--mixed-precision",
        mixed_precision,
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
    ]
    # Seed Python hashing before launch too: older Gemma 4 implementations register
    # RoPE buffers from a set, while DDP broadcasts buffers in registration order.
    with patch_environment(
        omp_num_threads=1, cublas_workspace_config=":4096:8", hf_hub_offline="1", pythonhashseed="0"
    ):
        execute_subprocess_async(
            [sys.executable, script_path, "--reference", "--output", str(reference_results_path)] + training_args,
            timeout=90,
        )
        execute_subprocess_async(
            launch_command + [script_path, "--output", str(ddp_results_path)] + training_args, timeout=90
        )

    reference_results, ddp_results = (
        json.loads(reference_results_path.read_text()),
        json.loads(ddp_results_path.read_text()),
    )
    assert reference_results["world_size"] == 1
    assert ddp_results["world_size"] == 2
    expected_dtype = {"no": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16"}[mixed_precision]
    expected_skips = [False] * 10
    if mixed_precision == "fp16":
        expected_skips[1] = True
    for result in (reference_results, ddp_results):
        assert len(result["losses"]) == 10
        assert result["parameters"]
        assert len(result["ranks"]) == result["world_size"]
        for rank_results in result["ranks"]:
            assert rank_results["parameter_dtypes"] == ["torch.float32"]
            assert rank_results["linear_output_dtypes"] == [expected_dtype]
            assert rank_results["step_was_skipped"] == expected_skips
            assert all(rank_results["parameters_changed"][i] for i in range(10) if not expected_skips[i])
            if mixed_precision == "fp16":
                assert rank_results["loss_scales"] == [128.0] + [64.0] * 9
                assert not rank_results["parameters_changed"][1]
    for rank_results in ddp_results["ranks"]:
        assert len(rank_results["parameters_unchanged_during_accumulation"]) == 10 * (gradient_accumulation_steps - 1)
        assert all(rank_results["parameters_unchanged_during_accumulation"])
    reference_parameters = torch.tensor(reference_results["parameters"], dtype=torch.float64)
    ddp_parameters = torch.tensor(ddp_results["parameters"], dtype=torch.float64)
    # Absolute bounds can hide incorrectly scaled small updates. Normalize the
    # final-parameter error by the reference's net movement from initialization.
    assert reference_results["parameter_delta_norm"] > 0
    relative_update_error = (reference_parameters - ddp_parameters).norm().item() / reference_results[
        "parameter_delta_norm"
    ]
    assert relative_update_error < 0.15, f"Relative update error: {relative_update_error:.3%}"
    # Different batch and reduction orders introduce rounding, especially under AMP.
    for field, atol in (("losses", loss_atol), ("parameters", parameter_atol)):
        torch.testing.assert_close(
            torch.tensor(reference_results[field], dtype=torch.float64),
            torch.tensor(ddp_results[field], dtype=torch.float64),
            rtol=0,
            atol=atol,
            msg=lambda message,
            field=field: f"{mixed_precision}, accumulation={gradient_accumulation_steps}, {field}: {message}",
        )
