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

from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from accelerate.test_utils.distributed_training import run_token_weighting_example, run_training
from accelerate.test_utils.testing import (
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available


DDP_CONFIG_FILE = Path(__file__).parent / "ddp.yaml"


@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training(tmp_path):
    """
    Train on eight examples per update: one GPU with eight, or two GPUs with four each.
    Compare the losses and require both runs to improve on the first batch.
    A plain-PyTorch reference can expose wrapper bugs that two Accelerate runs could share.
    """
    reference = run_training(
        tmp_path / "reference.json",
        reference=True,
        batch_size=8,
        mixed_precision="no",
    )
    distributed = run_training(
        tmp_path / "ddp.json",
        config_file=DDP_CONFIG_FILE,
        batch_size=4,
        mixed_precision="no",
    )

    max_loss_difference = 1e-4
    min_loss_decrease = 1e-4

    assert reference["world_size"] == 1
    assert distributed["world_size"] == 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10

    assert_close(distributed["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    assert_close(distributed["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement alone also accepts two runs that never learn. Require progress on the first batch.
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert distributed["final_loss"] < distributed["losses"][0] - min_loss_decrease


@pytest.mark.parametrize(
    "mixed_precision, max_loss_difference, min_loss_decrease",
    [
        pytest.param("fp16", 1e-4, 1e-4, id="fp16"),
        pytest.param(
            "bf16", 1e-3, 1e-3, id="bf16", marks=pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
        ),
    ],
)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_mixed_precision(tmp_path, mixed_precision, max_loss_difference, min_loss_decrease):
    """Keep eight examples per update and use the same requested precision in both runs.

    Compare the losses and require progress; the worker also checks that mixed precision ran.
    """
    reference = run_training(
        tmp_path / "reference.json",
        reference=True,
        batch_size=8,
        mixed_precision=mixed_precision,
    )
    distributed = run_training(
        tmp_path / "ddp.json",
        config_file=DDP_CONFIG_FILE,
        batch_size=4,
        mixed_precision=mixed_precision,
    )

    assert reference["world_size"] == 1
    assert distributed["world_size"] == 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10

    assert_close(distributed["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    assert_close(distributed["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement alone also accepts two runs that never learn. Require progress on the first batch.
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert distributed["final_loss"] < distributed["losses"][0] - min_loss_decrease


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path):
    """Keep BF16 and eight examples per update across two GPUs.

    Compare four examples per GPU at once with two batches of two examples per GPU.
    Their losses should agree, and both runs should improve on the first batch.
    """
    large_batch = run_training(
        tmp_path / "large.json",
        config_file=DDP_CONFIG_FILE,
        batch_size=4,
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
    )
    accumulated = run_training(
        tmp_path / "accumulated.json",
        config_file=DDP_CONFIG_FILE,
        batch_size=2,
        mixed_precision="bf16",
        gradient_accumulation_steps=2,
    )

    max_loss_difference = 1e-3
    min_loss_decrease = 1e-3

    assert large_batch["world_size"] == accumulated["world_size"] == 2
    assert len(large_batch["losses"]) == len(accumulated["losses"]) == 10

    assert_close(accumulated["losses"], large_batch["losses"], atol=max_loss_difference, rtol=0)
    assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement alone also accepts two runs that never learn. Require progress on the first batch.
    assert large_batch["final_loss"] < large_batch["losses"][0] - min_loss_decrease
    assert accumulated["final_loss"] < accumulated["losses"][0] - min_loss_decrease


@require_huggingface_suite
@pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()), reason="Requires Gloo"
)
def test_gradient_accumulation_example(tmp_path):
    """Starting from the same model and the same examples, does dividing training into smaller batches
    with unequal numbers of prediction targets produce the expected update?

    Exercise the actual example's training loop on two CPU ranks, including in CPU-only CI.
    The example suite separately checks its command-line entry point and collators.
    """
    results = run_token_weighting_example(
        tmp_path / "token_weighting.json",
        example_file=Path("examples/by_feature/gradient_accumulation_for_autoregressive_models.py").resolve(),
        num_processes=2,
    )

    expected_target_counts = [[3, 7], [12, 6]]

    for rank, result in enumerate(results):
        # Did this rank receive the intended targets and take one step?
        assert result["target_counts"] == expected_target_counts[rank]
        assert len(result["gradients"]) == 1

        # Do the gradients match? AdamW can hide a scaling error in its update.
        gradients = torch.tensor(result["gradients"][0], dtype=torch.float64)
        reference_gradients = torch.tensor(result["reference_gradients"], dtype=torch.float64)
        reference_gradient_norm = reference_gradients.norm().item()
        assert reference_gradient_norm > 0

        relative_gradient_error = (gradients - reference_gradients).norm().item() / reference_gradient_norm
        assert relative_gradient_error < 1e-4, (
            f"Rank {rank}: Token-weighted relative gradient error: {relative_gradient_error:.6%}"
        )

        # Do the updates match relative to the size of the reference update?
        parameters = torch.tensor(result["parameters"], dtype=torch.float64)
        reference_parameters = torch.tensor(result["reference_parameters"], dtype=torch.float64)
        reference_update_norm = result["reference_update_norm"]
        assert reference_update_norm > 0

        relative_update_error = (parameters - reference_parameters).norm().item() / reference_update_norm
        assert relative_update_error < 1e-4, (
            f"Rank {rank}: Token-weighted relative update error: {relative_update_error:.6%}"
        )
