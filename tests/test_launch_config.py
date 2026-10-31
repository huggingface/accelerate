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
import os

import pytest
import yaml

from accelerate.commands.launch import _validate_launch_command, launch_command_parser
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils.launch import prepare_simple_launcher_cmd_env


@pytest.fixture(params=["json", "yaml"])
def config_file(request, tmp_path):
    def write(**values):
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "mixed_precision": "no",
            "use_cpu": True,
            "num_processes": 1,
            "num_machines": 1,
            "debug": False,
        }
        config.update(values)
        path = tmp_path / f"config.{request.param}"
        path.write_text(json.dumps(config) if request.param == "json" else yaml.safe_dump(config))
        return path

    return write


def parse_config(path, *options):
    return launch_command_parser().parse_args(
        ["--config_file", str(path), "--num_cpu_threads_per_process", "1", *options, "train.py"]
    )


@pytest.mark.parametrize("name", ["same_network", "enable_cpu_affinity"])
@pytest.mark.parametrize("value", [True, False, None])
def test_top_level_boolean_config(config_file, name, value):
    path = config_file(**({} if value is None else {name: value}))
    original = path.read_bytes()
    args, defaults, _ = _validate_launch_command(parse_config(path))

    assert getattr(args, name) is (False if value is None else value)
    assert getattr(defaults, name) is (False if value is None else value)
    assert path.read_bytes() == original


@pytest.mark.parametrize("name", ["same_network", "enable_cpu_affinity"])
@pytest.mark.parametrize("separator", ["_", "-"])
def test_explicit_boolean_cli_overrides_config(config_file, name, separator):
    path = config_file(**{name: False})
    args, _, _ = _validate_launch_command(parse_config(path, "--" + name.replace("_", separator)))
    assert getattr(args, name) is True


@pytest.mark.parametrize("name", ["same_network", "enable_cpu_affinity"])
@pytest.mark.parametrize("value", [True, False])
def test_namespace_without_nondefault_retains_boolean(config_file, name, value):
    path = config_file(**{name: not value})
    args = parse_config(path)
    del args.nondefault
    setattr(args, name, value)
    args, _, _ = _validate_launch_command(args)
    assert getattr(args, name) is value


@pytest.mark.parametrize("enabled", [True, False])
def test_config_cpu_affinity_reaches_launcher_environment(config_file, monkeypatch, enabled):
    monkeypatch.delenv("ACCELERATE_CPU_AFFINITY", raising=False)
    path = config_file(enable_cpu_affinity=enabled, same_network=enabled)
    args, _, _ = _validate_launch_command(parse_config(path))
    original_environment = os.environ.copy()
    command, environment = prepare_simple_launcher_cmd_env(args)

    assert command[-1] == "train.py"
    affinity = environment.get("ACCELERATE_CPU_AFFINITY")
    assert affinity == ("1" if enabled else None)
    assert args.same_network is enabled
    assert os.environ == original_environment


@pytest.mark.parametrize("explicit", [False, True])
def test_no_config_boolean_defaults(explicit):
    options = ["--same_network", "--enable_cpu_affinity"] if explicit else []
    args = launch_command_parser().parse_args(
        ["--cpu", "--num_processes", "1", "--num_cpu_threads_per_process", "1", *options, "train.py"]
    )
    args, defaults, _ = _validate_launch_command(args)
    assert defaults is None
    assert args.same_network is explicit
    assert args.enable_cpu_affinity is explicit


@pytest.mark.parametrize("explicit", [False, True])
def test_scalar_and_nested_config_precedence(config_file, explicit):
    path = config_file(num_processes=2, dynamo_config={"dynamo_backend": "EAGER"}, same_network=True)
    options = ["--num_processes", "3", "--dynamo_backend", "no"] if explicit else []
    args, _, _ = _validate_launch_command(parse_config(path, *options))
    assert args.num_processes == (3 if explicit else 2)
    assert args.dynamo_backend == ("no" if explicit else "EAGER")


@pytest.mark.parametrize("name", ["same_network", "enable_cpu_affinity"])
@pytest.mark.parametrize("explicit", [False, True])
def test_launch_subcommand_boolean_precedence(config_file, name, explicit):
    path = config_file(**{name: not explicit})
    parser = CustomArgumentParser()
    launch_command_parser(subparsers=parser.add_subparsers())
    options = ["--" + name] if explicit else []
    args = parser.parse_args(
        ["launch", "--config_file", str(path), "--num_cpu_threads_per_process", "1", *options, "train.py"]
    )
    args, _, _ = _validate_launch_command(args)
    assert getattr(args, name) is True


@pytest.mark.parametrize("name", ["same_network", "enable_cpu_affinity"])
def test_default_config_boolean(config_file, monkeypatch, name):
    path = config_file(**{name: True})
    monkeypatch.setattr("accelerate.commands.launch.default_config_file", str(path))
    monkeypatch.setattr("accelerate.commands.config.config_args.default_config_file", str(path))
    args = launch_command_parser().parse_args(["--num_cpu_threads_per_process", "1", "train.py"])
    args, _, _ = _validate_launch_command(args)
    assert getattr(args, name) is True
