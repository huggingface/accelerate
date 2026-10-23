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

"""
Checks shared by the DDP, FSDP and DeepSpeed tests.

Each check trains a model on 2 GPUs with a distributed setup and compares it with a reference: the same training with
`Accelerator` on a single GPU, without any distributed wrapper. `train.py` always trains for 8 optimizer steps of 8
samples, so a distributed setup that trains correctly gives the same losses as the reference.
"""

import functools
import json
import os
import sys
import tempfile
import unittest

from torch.testing import assert_close

from accelerate.test_utils.testing import execute_subprocess_async, get_launch_command, get_torch_dist_unique_port


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "accelerate_configs")
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "train.py")


def run_script(script, config_file=None, **script_args):
    """
    Run `script` with `accelerate launch` and the accelerate `config_file`, or on a single GPU with `python` if there is
    none. `script_args` are passed to the script as `--key=value`.
    """
    if config_file:
        cmd = get_launch_command(config_file=config_file, main_process_port=get_torch_dist_unique_port())
    else:
        cmd = [sys.executable]
    cmd += [script] + [f"--{key}={value}" for key, value in script_args.items()]
    execute_subprocess_async(cmd)


@functools.cache
def train(config_file=None, output_dir=None, **script_args):
    """
    Run `train.py` and return the loss of each optimizer step. Runs are cached, so a run shared by several checks, like
    a reference, only happens once.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = output_dir or tmp_dir
        run_script(TRAIN_SCRIPT, config_file, output_dir=output_dir, **script_args)
        with open(os.path.join(output_dir, "losses.json")) as f:
            return json.load(f)


class DistributedTrainingCommon(unittest.TestCase):
    def assert_same_training(self, losses, reference_losses, atol=1e-3):
        """
        The loss should go down, and match the reference. Bugs such as a wrong loss scaling or a resume that doesn't
        restore the optimizer move the loss by more than 5e-2.
        """
        self.assertLess(losses[-1], losses[0])
        assert_close(losses, reference_losses, rtol=0, atol=atol)

    def check_training(self, dtype, config_file, atol=1e-3):
        """Training with the model loaded in `dtype` should match the reference."""
        reference = train(dtype=dtype)
        distributed = train(config_file, dtype=dtype)
        self.assert_same_training(distributed, reference, atol=atol)

    def check_mixed_precision(self, mixed_precision, config_file, atol=1e-3):
        """Training with the model in fp32 and the compute in `mixed_precision` should match the reference."""
        reference = train(mixed_precision=mixed_precision)
        distributed = train(config_file, mixed_precision=mixed_precision)
        self.assert_same_training(distributed, reference, atol=atol)

    def check_gradient_accumulation(self, config_file, dtype="bf16"):
        """Accumulating 4 steps of 1 sample per GPU should match 1 step of 4 samples per GPU, and the reference."""
        reference = train(dtype=dtype)
        distributed = train(config_file, dtype=dtype)
        accumulated = train(config_file, dtype=dtype, gradient_accumulation_steps=4)
        self.assert_same_training(accumulated, distributed)
        self.assert_same_training(accumulated, reference)

    def check_compiled_training(self, dtype, config_file):
        """Training with `torch.compile` should match the reference, and compile each block without graph breaks."""
        reference = train(dtype=dtype)
        with tempfile.TemporaryDirectory() as output_dir:
            compiled = train(config_file, output_dir=output_dir, dtype=dtype)
            with open(os.path.join(output_dir, "compile_stats.json")) as f:
                compile_stats = json.load(f)
        self.assert_same_training(compiled, reference)
        self.assertGreater(compile_stats["compiled_graphs"], 0)
        self.assertEqual(compile_stats["graph_breaks"], {})

    def check_resume(self, config_file):
        """Resuming halfway, from the checkpoint of step 4 out of 8, should match the training that didn't stop."""
        with tempfile.TemporaryDirectory() as output_dir:
            full = train(config_file, output_dir=output_dir, dtype="bf16", save_steps=4)
            checkpoint = os.path.join(output_dir, "step_4")
            resumed = train(config_file, dtype="bf16", resume_from_checkpoint=checkpoint)
        self.assert_same_training(resumed, full[4:])
