# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import subprocess
import unittest

import pytest

from accelerate.commands.launch import (
    CHILD_STDERR_CHUNK_SIZE,
    CHILD_STDERR_TAIL_CHUNKS,
    ChildProcessFailure,
    launch_command_parser,
    simple_launcher,
)
from accelerate.utils.launch import prepare_multi_gpu_env


class TestPrepareMultiGpuEnv(unittest.TestCase):
    def test_auto_port_selection(self):
        args = argparse.Namespace(
            num_processes=1,
            num_machines=1,
            main_process_ip="127.0.0.1",
            main_process_port=0,
            machine_rank=0,
            module=False,
            no_python=False,
            debug=False,
            gpu_ids="all",
            mixed_precision="no",
            dynamo_backend="NO",
            dynamo_mode="default",
            dynamo_use_fullgraph=False,
            dynamo_use_dynamic=False,
            dynamo_use_regional_compilation=False,
            use_fsdp=False,
            fsdp_cpu_ram_efficient_loading=False,
            fsdp_sync_module_states=False,
            fsdp_version=None,
            fsdp_sharding_strategy=None,
            fsdp_reshard_after_forward=False,
            fsdp_offload_params=False,
            fsdp_min_num_params=0,
            fsdp_auto_wrap_policy=None,
            fsdp_transformer_layer_cls_to_wrap=None,
            fsdp_backward_prefetch=None,
            fsdp_state_dict_type=None,
            fsdp_forward_prefetch=False,
            fsdp_use_orig_params=False,
            fsdp_activation_checkpointing=False,
            use_tp=False,
            tp_size=1,
            use_megatron_lm=False,
            megatron_lm_tp_degree=1,
            megatron_lm_pp_degree=1,
            megatron_lm_gradient_clipping=1.0,
            megatron_lm_num_micro_batches=None,
            megatron_lm_sequence_parallelism=None,
            megatron_lm_recompute_activations=None,
            megatron_lm_use_distributed_optimizer=None,
            num_cpu_threads_per_process=1,
            enable_cpu_affinity=False,
            same_network=False,
            use_parallelism_config=False,
        )

        prepare_multi_gpu_env(args)
        self.assertIn("master_port", args.__dict__)
        self.assertNotEqual(args.master_port, "0")
        self.assertTrue(args.master_port.isdigit())


def _simple_launcher_args(script, quiet=False):
    # Spelled out rather than relying on the defaults `launch_command` fills in before dispatching.
    argv = [
        "--cpu",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        "1",
    ]
    if quiet:
        argv.append("--quiet")
    return launch_command_parser().parse_args([*argv, script])


class TestSimpleLauncher:
    """`simple_launcher` must surface the child's failure without withholding its output."""

    def _write(self, tmp_path, body):
        script = tmp_path / "child.py"
        script.write_text(body)
        return str(script)

    def test_child_stderr_is_written_through_and_attached(self, tmp_path, capfd):
        script = self._write(
            tmp_path, "import sys\nprint('progress', file=sys.stderr)\nraise ValueError('the real cause')\n"
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            simple_launcher(_simple_launcher_args(script))

        # The child's output still reaches the terminal, as it did when stderr was inherited.
        assert "the real cause" in capfd.readouterr().err
        # And the caller can reach it, both as an attribute and as the chained cause.
        assert "the real cause" in exc_info.value.stderr
        assert isinstance(exc_info.value.__cause__, ChildProcessFailure)
        assert "the real cause" in str(exc_info.value.__cause__)

    def test_successful_child_output_is_not_withheld(self, tmp_path, capfd):
        script = self._write(tmp_path, "import sys\nprint('all good', file=sys.stderr)\n")
        simple_launcher(_simple_launcher_args(script))
        assert "all good" in capfd.readouterr().err

    def test_quiet_exits_without_raising(self, tmp_path):
        script = self._write(tmp_path, "raise ValueError('the real cause')\n")
        with pytest.raises(SystemExit) as exc_info:
            simple_launcher(_simple_launcher_args(script, quiet=True))
        assert exc_info.value.code == 1

    def test_tail_is_bounded_for_a_noisy_child(self, tmp_path, capfd):
        script = self._write(
            tmp_path,
            "import sys\nfor i in range(200_000):\n    print('x' * 40, file=sys.stderr)\nraise ValueError('the real cause')\n",
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            simple_launcher(_simple_launcher_args(script))

        # The child floods stderr, which would deadlock a wait()-then-read() launcher and blow up a communicate() one.
        assert "the real cause" in exc_info.value.stderr
        assert len(exc_info.value.stderr) <= CHILD_STDERR_CHUNK_SIZE * CHILD_STDERR_TAIL_CHUNKS
        assert len(capfd.readouterr().err) > CHILD_STDERR_CHUNK_SIZE * CHILD_STDERR_TAIL_CHUNKS
