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
import os
import subprocess
import unittest

import pytest

from accelerate.commands.launch import (
    CHILD_STDERR_CHUNK_SIZE,
    CHILD_STDERR_TAIL_CHUNKS,
    launch_command_parser,
    simple_launcher,
)
from accelerate.launchers import debug_launcher, notebook_launcher
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
    def test_child_stderr_is_written_through_and_attached(self, tmp_path, capfd):
        script = tmp_path / "child.py"
        # Floods stderr before failing: the case that deadlocks a wait()-then-read launcher and buffers
        # without limit in one that drains with communicate().
        script.write_text(
            "import sys\n"
            "for _ in range(200_000):\n"
            "    print('x' * 40, file=sys.stderr)\n"
            "raise ValueError('the real cause')\n"
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            simple_launcher(_simple_launcher_args(str(script)))

        # The child's output still reaches the terminal in full, as it did when stderr was inherited.
        captured = capfd.readouterr().err
        assert "the real cause" in captured
        assert len(captured) > CHILD_STDERR_CHUNK_SIZE * CHILD_STDERR_TAIL_CHUNKS
        # The caller can reach the cause, and what is retained for it stays bounded.
        assert "the real cause" in exc_info.value.stderr
        assert len(exc_info.value.stderr) <= CHILD_STDERR_CHUNK_SIZE * CHILD_STDERR_TAIL_CHUNKS
        assert "the real cause" in str(exc_info.value.__cause__)


def test_notebook_launcher_sets_accelerate_mixed_precision(monkeypatch):
    # notebook_launcher used to set a bare MIXED_PRECISION key, which nothing
    # in accelerate reads; the workers read ACCELERATE_MIXED_PRECISION.
    captured = {}
    monkeypatch.setattr(
        "torch.distributed.launcher.api.elastic_launch",
        lambda config, entrypoint: lambda *a: captured.update(
            accel=os.environ.get("ACCELERATE_MIXED_PRECISION"), bare=os.environ.get("MIXED_PRECISION")
        ),
    )
    notebook_launcher(lambda: None, num_processes=2, mixed_precision="fp16", use_port="29613")
    assert captured["accel"] == "fp16"
    assert captured["bare"] is None


def test_notebook_launcher_invalid_precision_error():
    with pytest.raises(ValueError, match="Unknown mixed_precision mode"):
        notebook_launcher(lambda: None, num_processes=1, mixed_precision="bogus")


def _run_debug_launcher(monkeypatch, probe=None, start_methods=("fork", "spawn"), interfaces=(("1", "lo"),)):
    """Run `debug_launcher` with the real workers replaced, and report what it would have started them with."""
    captured = {}

    def fake_start_processes(launcher, args=(), nprocs=1, start_method="fork"):
        captured["start_method"] = start_method
        captured["rdv_file"] = os.environ.get("ACCELERATE_DEBUG_RDV_FILE")
        captured["gloo_socket_ifname"] = os.environ.get("GLOO_SOCKET_IFNAME")
        captured["probe"] = None if probe is None else probe(captured["rdv_file"])

    monkeypatch.setattr("torch.multiprocessing.get_all_start_methods", lambda: list(start_methods))
    monkeypatch.setattr("torch.multiprocessing.start_processes", fake_start_processes)
    monkeypatch.setattr("socket.if_nameindex", lambda: list(interfaces))
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    debug_launcher(lambda: None, num_processes=2)
    return captured


def test_debug_launcher_uses_an_available_start_method(monkeypatch):
    # Windows only offers `spawn`: asking for `fork` there raises `cannot find context for 'fork'` before any
    # worker starts, which made `debug_launcher` unusable outside Unix-like platforms.
    assert _run_debug_launcher(monkeypatch)["start_method"] == "fork"
    assert _run_debug_launcher(monkeypatch, start_methods=("spawn",))["start_method"] == "spawn"


def test_debug_launcher_only_pins_loopback_where_it_exists(monkeypatch):
    # `lo` is the Unix name for the loopback interface. Pinning it where that name does not exist leaves gloo
    # unable to find any address at all.
    assert _run_debug_launcher(monkeypatch)["gloo_socket_ifname"] == "lo"
    assert _run_debug_launcher(monkeypatch, interfaces=(("1", "loopback_0"),))["gloo_socket_ifname"] is None


def test_debug_launcher_rendezvous_file_can_be_opened_by_a_worker(monkeypatch):
    # A `NamedTemporaryFile` stays open in this process for the whole launch. On Windows a spawned worker cannot
    # open a file the parent still holds open, so the path handed over has to be usable by the workers themselves.
    def write_then_read(rdv_file):
        with open(rdv_file, "w") as f:
            f.write("ok")
        with open(rdv_file) as f:
            return f.read()

    assert _run_debug_launcher(monkeypatch, probe=write_then_read)["probe"] == "ok"
