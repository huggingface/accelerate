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

from accelerate.test_utils.distributed_training import run_training
from accelerate.test_utils.testing import (
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available


DDP_CONFIG_FILE = Path(__file__).with_name("ddp.yaml")


@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training(tmp_path):
    """
    Compare DDP losses with single-GPU training on the same effective batches.
    A plain-PyTorch baseline can expose wrapper bugs that two Accelerate runs could share.
    """
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    distributed = run_training(tmp_path / "ddp.json", config_file=DDP_CONFIG_FILE, batch_size=4)

    max_loss_difference = 1e-4
    min_loss_decrease = 1e-4
    assert reference["world_size"] == 1
    assert distributed["world_size"] == 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)

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
    """Compare DDP with single-GPU training at the same requested precision."""
    reference = run_training(
        tmp_path / "reference.json", reference=True, batch_size=8, mixed_precision=mixed_precision
    )
    distributed = run_training(
        tmp_path / "ddp.json", config_file=DDP_CONFIG_FILE, batch_size=4, mixed_precision=mixed_precision
    )

    assert reference["world_size"] == 1
    assert distributed["world_size"] == 2
    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement alone also accepts two runs that never learn. Require progress on the first batch.
    assert reference["final_loss"] < reference["losses"][0] - min_loss_decrease
    assert distributed["final_loss"] < distributed["losses"][0] - min_loss_decrease


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path):
    """Keep BF16 and eight blocks per update: 2 ranks * 4 blocks, or 2 ranks * 2 blocks * 2 steps."""
    large_batch = run_training(
        tmp_path / "large.json", config_file=DDP_CONFIG_FILE, batch_size=4, mixed_precision="bf16"
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
    torch.testing.assert_close(accumulated["losses"], large_batch["losses"], atol=max_loss_difference, rtol=0)
    torch.testing.assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=max_loss_difference, rtol=0)

    # Agreement alone also accepts two runs that never learn. Require progress on the first batch.
    assert large_batch["final_loss"] < large_batch["losses"][0] - min_loss_decrease
    assert accumulated["final_loss"] < accumulated["losses"][0] - min_loss_decrease
