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

"""Check the actual autoregressive example's accumulation against a full-batch update."""

import os
import sys
from pathlib import Path

import pytest
import torch

from accelerate.test_utils.testing import (
    execute_subprocess_async,
    path_in_accelerate_package,
    require_huggingface_suite,
)


def run_token_weighting_example(output, *, example_file, num_processes):
    """Run the example's controlled CPU experiment and return each rank's measurements."""
    # Use this interpreter's torchrun to start communicating CPU processes.
    # Plain accelerate launch --cpu starts only one process.
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={num_processes}",
        path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_token_weighting.py"),
        "--example",
        example_file,
        "--output",
        output,
    ]
    # Keep the CPU fixture independent of the parent test environment.
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        # The example otherwise forces two epochs when this test flag is enabled.
        "TESTING_MOCKED_DATALOADERS": "0",
    }
    run_result = execute_subprocess_async(command, env=env)
    assert run_result.returncode == 0, f"DDP launcher failed: {run_result.stderr}"

    return [torch.load(output.with_suffix(f".rank{rank}.pt"), weights_only=True) for rank in range(num_processes)]


@require_huggingface_suite
@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()), reason="Requires Gloo"
)
def test_gradient_accumulation_example(tmp_path):
    """Starting from the same model and the same examples, does dividing training into smaller batches
    with unequal numbers of prediction targets produce the expected update?

    Two CPU processes exercise global target counting and gradient averaging.
    No GPU-specific behavior is needed for this normalization check.
    The example suite separately checks its command-line entry point and collators.
    """
    results = run_token_weighting_example(
        tmp_path / "token_weighting.pt",
        example_file=Path("examples/by_feature/gradient_accumulation_for_autoregressive_models.py").resolve(),
        num_processes=2,
    )

    expected_target_counts = [[3, 7], [12, 6]]

    for rank, result in enumerate(results):
        # Did this rank receive the intended targets and take one step?
        assert result["target_counts"] == expected_target_counts[rank]
        assert len(result["gradients"]) == 1

        # Do the gradients match? AdamW can hide a scaling error in its update.
        gradients = result["gradients"][0].to(torch.float64)
        reference_gradients = result["reference_gradients"].to(torch.float64)
        reference_gradient_norm = reference_gradients.norm().item()
        assert reference_gradient_norm > 0

        relative_gradient_error = (gradients - reference_gradients).norm().item() / reference_gradient_norm
        assert relative_gradient_error < 1e-4, (
            f"Rank {rank}: Token-weighted relative gradient error: {relative_gradient_error:.6%}"
        )

        # Do the updates match relative to the size of the reference update?
        parameters = result["parameters"].to(torch.float64)
        reference_parameters = result["reference_parameters"].to(torch.float64)
        reference_update_norm = result["reference_update_norm"]
        assert reference_update_norm > 0

        relative_update_error = (parameters - reference_parameters).norm().item() / reference_update_norm
        assert relative_update_error < 1e-4, (
            f"Rank {rank}: Token-weighted relative update error: {relative_update_error:.6%}"
        )
