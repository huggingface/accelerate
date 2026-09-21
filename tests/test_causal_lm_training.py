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
from accelerate.test_utils.scripts.external_deps.train_causal_lm import (
    INITIAL_LOSS_SCALE,
    LOSS_SCALE_BACKOFF,
    NUM_UPDATE_ATTEMPTS,
    microbatch_size_for,
    parse_args,
)
from accelerate.test_utils.testing import (
    execute_subprocess_async,
    get_launch_command,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
    require_cuda,
    require_multi_gpu,
    require_transformers,
)
from accelerate.utils import ComputeEnvironment, DistributedType, patch_environment


@require_cuda
@require_multi_gpu
@require_transformers
@pytest.mark.parametrize(
    "mixed_precision, gradient_accumulation_steps, loss_atol, parameter_atol",
    [
        # Does distributing FP32 training work without gradient accumulation?
        pytest.param("no", 1, 1e-4, 1e-5, id="fp32_ddp"),
        # Does it still work when we add gradient accumulation?
        pytest.param("no", 2, 1e-4, 1e-5, id="fp32_accumulation"),
        # Does accumulation work with BF16 mixed precision?
        pytest.param("bf16", 2, 1e-2, 1e-3, id="bf16_accumulation"),
        # Does accumulation work with FP16 gradient scaling, without injected faults?
        pytest.param("fp16", 2, 1e-3, 1e-4, id="fp16_accumulation"),
    ],
)
def test_ddp_training_matches_reference(
    tmp_path,
    mixed_precision,
    gradient_accumulation_steps,
    loss_atol,
    parameter_atol,
):
    """DDP, accumulation and AMP match a same-precision plain-PyTorch reference."""
    # Given: identical starting weights/data, the selected precision, and no injected fault.
    _require_precision_support(mixed_precision)

    # When: train full reference batches and equivalent distributed, accumulated batches.
    reference, ddp = _run_training_pair(tmp_path, mixed_precision, gradient_accumulation_steps)

    # Then: every update attempt succeeds, and both paths exercise the requested behavior and agree.
    _assert_execution_contract(reference, ddp, mixed_precision, gradient_accumulation_steps)
    expected_scale = INITIAL_LOSS_SCALE if mixed_precision == "fp16" else 1.0
    for context, attempt in _rank_attempts(reference, ddp):
        assert not attempt["step_was_skipped"], context
        assert attempt["parameters_changed"], context
        assert attempt["loss_scale"] == expected_scale, context
    _assert_numerical_parity(reference, ddp, loss_atol, parameter_atol)


@require_cuda
@require_multi_gpu
@require_transformers
def test_fp16_skips_nonfinite_update_and_recovers(tmp_path):
    """Reject a deliberately corrupted update, then resume matching the reference."""
    # Given: FP16 with accumulation, and an explicit fault at the second update attempt.
    _require_precision_support("fp16")
    gradient_accumulation_steps = 2
    fault_attempt = 1  # Zero-based: first establish a successful update, then test rejection and recovery.

    # When: inject infinity after backward at that boundary in both paths (on every DDP rank).
    reference, ddp = _run_training_pair(
        tmp_path, "fp16", gradient_accumulation_steps, inject_nonfinite_at_attempt=fault_attempt
    )

    # Then: only the faulty attempt is skipped; weights hold, the scale falls, and later updates succeed.
    _assert_execution_contract(reference, ddp, "fp16", gradient_accumulation_steps)
    for context, attempt in _rank_attempts(reference, ddp):
        expected_skip = attempt["index"] == fault_attempt
        expected_scale = INITIAL_LOSS_SCALE
        if attempt["index"] >= fault_attempt:
            expected_scale *= LOSS_SCALE_BACKOFF
        assert attempt["step_was_skipped"] == expected_skip, context
        assert attempt["parameters_changed"] == (not expected_skip), context
        assert attempt["loss_scale"] == expected_scale, context
    _assert_numerical_parity(reference, ddp, loss_atol=1e-3, parameter_atol=1e-4)


def _require_precision_support(mixed_precision):
    pytest.importorskip("transformers.models.gemma4", reason="Requires Transformers with Gemma 4 support")
    if mixed_precision == "bf16":
        for device in range(2):
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    pytest.skip("Both CUDA devices must support BF16")


def _run_training_pair(tmp_path, mixed_precision, gradient_accumulation_steps, inject_nonfinite_at_attempt=None):
    """Launch independent reference/DDP experiments with the same explicit scenario."""
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
    if inject_nonfinite_at_attempt is not None:
        training_args += ["--inject-nonfinite-at-attempt", str(inject_nonfinite_at_attempt)]

    with patch_environment(
        # Avoid competing OpenMP thread pools in the training processes.
        omp_num_threads=1,
        # Keep CUDA matrix multiplications repeatable for the numerical comparison.
        cublas_workspace_config=":4096:8",
        # Keep this synthetic test independent of Hub availability.
        hf_hub_offline="1",
        # Some Gemma implementations register RoPE buffers from a set.
        # Use matching hash seeds so DDP's positional broadcasts don't mix up buffers.
        pythonhashseed="0",
    ):
        reference_process = execute_subprocess_async(
            [sys.executable, script_path, "--reference", "--output", str(reference_results_path)] + training_args,
            timeout=90,
        )
        assert reference_process.returncode == 0, f"Reference process failed: {reference_process.stderr}"

        ddp_process = execute_subprocess_async(
            launch_command + [script_path, "--output", str(ddp_results_path)] + training_args,
            timeout=90,
        )
        assert ddp_process.returncode == 0, f"DDP launcher failed: {ddp_process.stderr}"

    return json.loads(reference_results_path.read_text()), json.loads(ddp_results_path.read_text())


def _rank_attempts(reference, ddp):
    for path, result in (("reference", reference), ("DDP", ddp)):
        for rank, observations in enumerate(result["ranks"]):
            for attempt in observations["attempts"]:
                yield f"{path}, rank {rank}, attempt {attempt['index']}", attempt


def _assert_execution_contract(reference, ddp, mixed_precision, gradient_accumulation_steps):
    """Check coverage of the requested execution, independently of numerical closeness."""
    assert reference["world_size"] == 1
    assert ddp["world_size"] == 2
    expected_dtype = {"no": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16"}[mixed_precision]
    for path, result in (("reference", reference), ("DDP", ddp)):
        assert result["parameters"], path
        assert len(result["ranks"]) == result["world_size"], path
        for rank, observations in enumerate(result["ranks"]):
            context = f"{path}, rank {rank}"
            assert [attempt["index"] for attempt in observations["attempts"]] == list(range(NUM_UPDATE_ATTEMPTS)), (
                context
            )
            assert observations["parameter_dtypes"] == ["torch.float32"], context
            assert observations["linear_output_dtypes"] == [expected_dtype], context

    for rank, observations in enumerate(ddp["ranks"]):
        holds = observations["parameters_unchanged_during_accumulation"]
        assert len(holds) == NUM_UPDATE_ATTEMPTS * (gradient_accumulation_steps - 1), f"DDP rank {rank}"
        for microbatch, unchanged in enumerate(holds):
            assert unchanged, f"DDP rank {rank}, accumulation interior {microbatch}: weights changed before boundary"


def _assert_numerical_parity(reference, ddp, loss_atol, parameter_atol):
    reference_parameters = torch.tensor(reference["parameters"], dtype=torch.float64)
    ddp_parameters = torch.tensor(ddp["parameters"], dtype=torch.float64)
    # Absolute bounds can hide incorrectly scaled small updates. Normalize the
    # final-parameter error by the reference's net movement from initialization.
    assert reference["parameter_delta_norm"] > 0
    relative_update_error = (reference_parameters - ddp_parameters).norm().item() / reference["parameter_delta_norm"]
    assert relative_update_error < 0.15, f"Relative update error: {relative_update_error:.3%}"

    # Different batch and reduction orders introduce rounding, especially under AMP.
    reference_losses = [attempt["loss"] for attempt in reference["ranks"][0]["attempts"]]
    ddp_losses = [attempt["loss"] for attempt in ddp["ranks"][0]["attempts"]]
    for field, actual, expected, atol in (
        (
            "losses",
            torch.tensor(ddp_losses, dtype=torch.float64),
            torch.tensor(reference_losses, dtype=torch.float64),
            loss_atol,
        ),
        ("parameters", ddp_parameters, reference_parameters, parameter_atol),
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=atol,
            msg=lambda message, field=field: f"{field}: {message}",
        )


def test_training_fault_is_opt_in():
    args = parse_args(["--output", "unused.json", "--mixed-precision", "fp16"])
    assert args.inject_nonfinite_at_attempt is None
    args = parse_args(["--output", "unused.json", "--mixed-precision", "fp16", "--inject-nonfinite-at-attempt", "1"])
    assert args.inject_nonfinite_at_attempt == 1


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--gradient-accumulation-steps", "0"], "accumulation steps must be positive"),
        (["--gradient-accumulation-steps", "-1"], "accumulation steps must be positive"),
        (["--inject-nonfinite-at-attempt", "1"], "injection requires FP16"),
        (["--mixed-precision", "bf16", "--inject-nonfinite-at-attempt", "1"], "injection requires FP16"),
        (["--mixed-precision", "fp16", "--inject-nonfinite-at-attempt", "-1"], "injection attempt must be between"),
        (
            ["--mixed-precision", "fp16", "--inject-nonfinite-at-attempt", str(NUM_UPDATE_ATTEMPTS)],
            "injection attempt must be between",
        ),
    ],
)
def test_invalid_training_scenario_is_rejected(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["--output", "unused.json", *arguments])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("accumulation, expected_microbatch_size", [(1, 4), (2, 2)])
def test_microbatch_layout_preserves_effective_batch(accumulation, expected_microbatch_size):
    assert microbatch_size_for(8, 2, accumulation) == expected_microbatch_size


@pytest.mark.parametrize("batch, ranks, accumulation", [(8, 2, 3), (8, 2, 8), (0, 2, 1), (8, 0, 1), (8, 2, 0)])
def test_invalid_microbatch_layout_is_rejected(batch, ranks, accumulation):
    with pytest.raises(ValueError, match="positive|divisible"):
        microbatch_size_for(batch, ranks, accumulation)
