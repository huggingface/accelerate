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
from pathlib import Path

import pytest
import torch

from accelerate.test_utils.testing import (
    execute_subprocess_async,
    get_torch_dist_unique_port,
    path_in_accelerate_package,
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
)
from accelerate.utils import is_bf16_available, patch_environment


@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training(tmp_path):
    """
    Compare DDP losses with single-GPU training on the same effective batches.
    A plain-PyTorch baseline can expose wrapper bugs that two Accelerate runs could share.
    """
    reference = run_training(tmp_path / "reference.json", reference=True, batch_size=8)
    distributed = run_training(tmp_path / "ddp.json", batch_size=4)

    atol = 1e-4
    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=atol, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=atol, rtol=0)
    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert reference["final_loss"] < reference["losses"][0] - atol
    assert distributed["final_loss"] < distributed["losses"][0] - atol


@pytest.mark.parametrize(
    "mixed_precision, atol",
    [
        pytest.param("fp16", 1e-4),
        pytest.param("bf16", 1e-3, marks=pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")),
    ],
)
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_mixed_precision(tmp_path, mixed_precision, atol):
    """Compare DDP with single-GPU training at the same requested precision."""
    reference = run_training(
        tmp_path / "reference.json", reference=True, batch_size=8, mixed_precision=mixed_precision
    )
    distributed = run_training(tmp_path / "ddp.json", batch_size=4, mixed_precision=mixed_precision)

    assert len(reference["losses"]) == len(distributed["losses"]) == 10
    torch.testing.assert_close(distributed["losses"], reference["losses"], atol=atol, rtol=0)
    torch.testing.assert_close(distributed["final_loss"], reference["final_loss"], atol=atol, rtol=0)
    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert reference["final_loss"] < reference["losses"][0] - atol
    assert distributed["final_loss"] < distributed["losses"][0] - atol


@pytest.mark.skipif(not is_bf16_available(), reason="Requires BF16")
@require_cuda
@require_multi_gpu
@require_huggingface_suite
def test_training_with_gradient_accumulation(tmp_path):
    """Keep BF16 and eight blocks per update: 2 ranks * 4 blocks, or 2 ranks * 2 blocks * 2 steps."""
    large_batch = run_training(tmp_path / "large.json", batch_size=4, mixed_precision="bf16")
    accumulated = run_training(
        tmp_path / "accumulated.json", batch_size=2, mixed_precision="bf16", gradient_accumulation_steps=2
    )

    atol = 1e-3
    assert len(large_batch["losses"]) == len(accumulated["losses"]) == 10
    torch.testing.assert_close(accumulated["losses"], large_batch["losses"], atol=atol, rtol=0)
    torch.testing.assert_close(accumulated["final_loss"], large_batch["final_loss"], atol=atol, rtol=0)
    # Agreement alone also accepts two runs that never learn. On the first batch,
    # require a loss decrease larger than the comparison tolerance.
    assert large_batch["final_loss"] < large_batch["losses"][0] - atol
    assert accumulated["final_loss"] < accumulated["losses"][0] - atol


def run_training(output, *, batch_size, mixed_precision="no", gradient_accumulation_steps=1, reference=False):
    script = path_in_accelerate_package("test_utils", "scripts", "external_deps", "train_causal_lm.py")
    command = [sys.executable]
    if not reference:
        command += [
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(Path(__file__).with_name("ddp.yaml")),
            "--main_process_port",
            str(get_torch_dist_unique_port()),
        ]
    command += [
        str(script),
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
    with patch_environment(omp_num_threads=1):
        result = execute_subprocess_async(command)
    assert result.returncode == 0, result.stderr
    result = json.loads(output.read_text())
    assert result["world_size"] == (1 if reference else 2)
    return result
