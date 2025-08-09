# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import argparse
import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils import (
    get_launch_command,
    require_cuda_or_hpu,
    require_huggingface_suite,
    require_multi_device,
    require_torchao,
    require_transformer_engine,
    run_first,
)
from accelerate.test_utils.testing import require_deepspeed, run_command
from accelerate.utils import (
    AORecipeKwargs,
    TERecipeKwargs,
    has_ao_layers,
    has_transformer_engine_layers,
)


def can_convert_te_model(from_config=False):
    if not from_config:
        accelerator_kwargs = {"mixed_precision": "fp8", "kwargs_handlers": [TERecipeKwargs()]}
    else:
        accelerator_kwargs = {}

    accelerator = Accelerator(**accelerator_kwargs)
    dataloader = torch.utils.data.DataLoader(torch.randn(10, 32), batch_size=2)
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    assert has_transformer_engine_layers(model)


def maintain_proper_deepspeed_config(expected_version):
    assert AcceleratorState().deepspeed_plugin.zero_stage == expected_version, (
        f"Expected zero stage {expected_version} but got {AcceleratorState().deepspeed_plugin.zero_stage}"
    )


def can_convert_ao_model(from_config=False):
    from transformers import AutoModelForSequenceClassification

    if not from_config:
        accelerator_kwargs = {"mixed_precision": "fp8", "kwargs_handlers": [AORecipeKwargs()]}
    else:
        accelerator_kwargs = {}

    accelerator = Accelerator(**accelerator_kwargs)
    dataloader = torch.utils.data.DataLoader(torch.randn(10, 32), batch_size=2)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    assert has_ao_layers(model)


@run_first
@require_transformer_engine
@require_cuda_or_hpu
class TestTransformerEngine(unittest.TestCase):
    def test_can_prepare_model_single_gpu(self):
        command = get_launch_command(num_processes=1, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8", "--test_te"]
        run_command(command)

    def test_can_prepare_model_single_gpu_from_config(self):
        with tempfile.TemporaryDirectory() as dir_name:
            config_file = Path(dir_name) / "config.yaml"
            config_file.write_text(
                textwrap.dedent(
                    """
                    distributed_type: "NO"
                    num_processes: 1
                    mixed_precision: fp8
                    fp8_config:
                      backend: TE
                    """
                )
            )
            command = get_launch_command(config_file=str(config_file), monitor_interval=0.1)
            command += ["-m", "tests.test_fp8", "--test_te", "--from_config"]
            run_command(command)

    @require_multi_device
    def test_can_prepare_model_multi_gpu(self):
        command = get_launch_command(num_processes=2, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8", "--test_te"]
        run_command(command)

    @require_deepspeed
    @require_multi_device
    def test_can_prepare_model_multigpu_deepspeed(self):
        for zero_stage in [1, 2, 3]:
            os.environ["ZERO_STAGE"] = str(zero_stage)
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": zero_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": 1,
                "gradient_clipping": "auto",
                "steps_per_print": 2000,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False,
            }

            ds_config = json.dumps(ds_config)

            command = get_launch_command(
                num_processes=2, monitor_interval=0.1, use_deepspeed=True, deepspeed_config_file=ds_config
            )
            command += ["-m", "tests.test_fp8", "--test_te"]
            run_command(command)


@require_torchao
@require_huggingface_suite
class TestTorchAO(unittest.TestCase):
    def test_can_prepare_model_single_accelerator(self):
        command = get_launch_command(num_processes=1, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8", "--test_ao"]
        run_command(command)

    def test_can_prepare_model_single_gpu_from_config(self):
        with tempfile.TemporaryDirectory() as dir_name:
            config_file = Path(dir_name) / "config.yaml"
            config_file.write_text(
                textwrap.dedent(
                    """
                    distributed_type: "NO"
                    num_processes: 1
                    mixed_precision: fp8
                    fp8_config:
                      backend: AO
                    """
                )
            )
            command = get_launch_command(config_file=str(config_file), monitor_interval=0.1)
            command += ["-m", "tests.test_fp8", "--test_ao", "--from_config"]
            run_command(command)

    @require_multi_device
    def test_can_prepare_model_multi_accelerator(self):
        command = get_launch_command(num_processes=2, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8", "--test_ao"]
        run_command(command)

    @require_deepspeed
    @require_multi_device
    def test_can_prepare_model_multi_accelerator_deepspeed(self):
        for zero_stage in [1, 2, 3]:
            os.environ["ZERO_STAGE"] = str(zero_stage)
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": zero_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": 1,
                "gradient_clipping": "auto",
                "steps_per_print": 2000,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False,
            }

            ds_config = json.dumps(ds_config)

            command = get_launch_command(
                num_processes=2, monitor_interval=0.1, use_deepspeed=True, deepspeed_config_file=ds_config
            )
            command += ["-m", "tests.test_fp8", "--test_ao"]
            run_command(command)


if __name__ == "__main__":
    # TE suite
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_te", action="store_true", default=False)
    parser.add_argument("--test_ao", action="store_true", default=False)
    parser.add_argument("--from_config", action="store_true", default=False)
    args = parser.parse_args()

    if not args.test_te and not args.test_ao:
        raise ValueError("Must specify at least one of --test_te or --test_ao")

    if args.test_te:
        can_convert_te_model(args.from_config)
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
            maintain_proper_deepspeed_config(int(os.environ.get("ZERO_STAGE")))

    # AO suite
    if args.test_ao:
        can_convert_ao_model(args.from_config)
