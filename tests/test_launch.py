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
import yaml

from accelerate.commands.launch import (
    CHILD_STDERR_CHUNK_SIZE,
    CHILD_STDERR_TAIL_CHUNKS,
    _validate_launch_command,
    launch_command_parser,
    simple_launcher,
)
from accelerate.launchers import notebook_launcher
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils.constants import FSDP_SHARDING_STRATEGY
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


@pytest.mark.parametrize("source", ["api", "environment", "cli", "config"])
@pytest.mark.parametrize("strategy", FSDP_SHARDING_STRATEGY)
def test_fsdp1_reshard_after_forward(tmp_path, monkeypatch, source, strategy):
    from torch.distributed.fsdp import ShardingStrategy

    monkeypatch.delenv("FSDP_SHARDING_STRATEGY", raising=False)
    monkeypatch.delenv("FSDP_RESHARD_AFTER_FORWARD", raising=False)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "false")
    kwargs = {"fsdp_version": 1, "sync_module_states": False, "cpu_ram_efficient_loading": False}
    if source == "api":
        kwargs["reshard_after_forward"] = strategy
    elif source == "environment":
        monkeypatch.setenv("FSDP_RESHARD_AFTER_FORWARD", strategy)
    else:
        config = {
            "distributed_type": "FSDP",
            "mixed_precision": "no",
            "num_processes": 2,
            "fsdp_config": {"fsdp_version": 1},
        }
        if source == "config":
            config["fsdp_config"]["fsdp_reshard_after_forward"] = strategy
        config_file = tmp_path / "accelerate.yaml"
        config_file.write_text(yaml.safe_dump(config))
        argv = ["--config_file", str(config_file)]
        if source == "cli":
            argv += ["--fsdp_reshard_after_forward", strategy]
        args, _, _ = _validate_launch_command(launch_command_parser().parse_args([*argv, "train.py"]))
        monkeypatch.setattr("accelerate.utils.launch.is_port_in_use", lambda port: False)
        for key, value in prepare_multi_gpu_env(args).items():
            monkeypatch.setenv(key, value)
    plugin = FullyShardedDataParallelPlugin(**kwargs)
    expected = ShardingStrategy(FSDP_SHARDING_STRATEGY.index(strategy) + 1)
    assert (plugin.sharding_strategy or plugin.reshard_after_forward) == expected


@pytest.mark.parametrize("legacy", ["SHARD_GRAD_OP", "2"])
@pytest.mark.parametrize("source", ["api", "environment"])
def test_fsdp1_explicit_legacy_strategy_takes_precedence(monkeypatch, source, legacy):
    from torch.distributed.fsdp import ShardingStrategy

    monkeypatch.delenv("FSDP_SHARDING_STRATEGY", raising=False)
    monkeypatch.delenv("FSDP_RESHARD_AFTER_FORWARD", raising=False)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "false")
    kwargs = {"fsdp_version": 1, "sync_module_states": False, "cpu_ram_efficient_loading": False}
    if source == "api":
        kwargs.update(sharding_strategy=legacy, reshard_after_forward="FULL_SHARD")
    else:
        monkeypatch.setenv("FSDP_SHARDING_STRATEGY", legacy)
        monkeypatch.setenv("FSDP_RESHARD_AFTER_FORWARD", "FULL_SHARD")
    plugin = FullyShardedDataParallelPlugin(**kwargs)
    assert (plugin.sharding_strategy or plugin.reshard_after_forward) == ShardingStrategy.SHARD_GRAD_OP


def test_fsdp_launch_keeps_strategy_defaults_unset(monkeypatch):
    monkeypatch.delenv("FSDP_SHARDING_STRATEGY", raising=False)
    monkeypatch.delenv("FSDP_RESHARD_AFTER_FORWARD", raising=False)
    monkeypatch.setattr("accelerate.utils.launch.is_port_in_use", lambda port: False)
    args = launch_command_parser().parse_args(
        [
            "--use_fsdp",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            "train.py",
        ]
    )
    env = prepare_multi_gpu_env(args)
    assert env.get("FSDP_SHARDING_STRATEGY") is None
    assert env.get("FSDP_RESHARD_AFTER_FORWARD") is None
    monkeypatch.setenv("FSDP_RESHARD_AFTER_FORWARD", "SHARD_GRAD_OP")
    assert prepare_multi_gpu_env(args)["FSDP_RESHARD_AFTER_FORWARD"] == "SHARD_GRAD_OP"
