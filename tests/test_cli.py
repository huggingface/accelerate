# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
import os
import unittest
from pathlib import Path

import torch

import accelerate
from accelerate.test_utils import execute_subprocess_async
from accelerate.test_utils.testing import run_command


class AccelerateLauncherTester(unittest.TestCase):
    """
    Test case for verifying the `accelerate launch` CLI operates correctly.
    If a `default_config.yaml` file is located in the cache it will temporarily move it
    for the duration of the tests.
    """

    mod_file = inspect.getfile(accelerate.test_utils)
    test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "test_cli.py"])

    base_cmd = ["accelerate", "launch"]
    config_folder = Path.home() / ".cache/huggingface/accelerate"
    config_file = "default_config.yaml"
    config_path = config_folder / config_file
    changed_path = config_folder / "_default_config.yaml"

    test_config_path = Path("tests/test_configs")

    @classmethod
    def setUpClass(cls):
        if cls.config_path.is_file():
            cls.config_path.rename(cls.changed_path)

    @classmethod
    def tearDownClass(cls):
        if cls.changed_path.is_file():
            cls.changed_path.rename(cls.config_path)

    def test_no_config(self):
        cmd = self.base_cmd
        if torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            cmd += ["--multi_gpu"]
        execute_subprocess_async(cmd + [self.test_file_path], env=os.environ.copy())

    def test_config_compatibility(self):
        for config in sorted(self.test_config_path.glob("**/*.yaml")):
            with self.subTest(config_file=config):
                execute_subprocess_async(
                    self.base_cmd + ["--config_file", str(config), self.test_file_path], env=os.environ.copy()
                )

    def test_accelerate_test(self):
        execute_subprocess_async(["accelerate", "test"], env=os.environ.copy())


class TpuConfigTester(unittest.TestCase):
    """
    Test case for verifying the `accelerate tpu-config` CLI passes the right `gcloud` command.
    """

    tpu_name = "test-tpu"
    tpu_zone = "us-central1-a"
    command = "ls"
    cmd = ["accelerate", "tpu-config"]
    base_output = "cd /usr/share"
    command_file = "tests/test_samples/test_command_file.sh"
    gcloud = "Running gcloud compute tpus tpu-vm ssh"

    @staticmethod
    def clean_output(output):
        return "".join(output).rstrip()

    def test_base(self):
        output = run_command(
            self.cmd
            + ["--command", self.command, "--tpu_zone", self.tpu_zone, "--tpu_name", self.tpu_name, "--debug"],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all",
        )

    def test_base_backward_compatibility(self):
        output = run_command(
            self.cmd
            + [
                "--config_file",
                "tests/test_configs/0_12_0.yaml",
                "--command",
                self.command,
                "--tpu_zone",
                self.tpu_zone,
                "--tpu_name",
                self.tpu_name,
                "--debug",
            ],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all",
        )

    def test_with_config_file(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--debug"], return_stdout=True
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all',
        )

    def test_with_config_file_and_command(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--command", self.command, "--debug"],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all",
        )

    def test_with_config_file_and_multiple_command(self):
        output = run_command(
            self.cmd
            + [
                "--config_file",
                "tests/test_configs/latest.yaml",
                "--command",
                self.command,
                "--command",
                'echo "Hello World"',
                "--debug",
            ],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls; echo "Hello World" --worker all',
        )

    def test_with_config_file_and_command_file(self):
        output = run_command(
            self.cmd
            + ["--config_file", "tests/test_configs/latest.yaml", "--command_file", self.command_file, "--debug"],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all',
        )

    def test_with_config_file_and_command_file_backward_compatibility(self):
        output = run_command(
            self.cmd
            + [
                "--config_file",
                "tests/test_configs/0_12_0.yaml",
                "--command_file",
                self.command_file,
                "--tpu_zone",
                self.tpu_zone,
                "--tpu_name",
                self.tpu_name,
                "--debug",
            ],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all',
        )

    def test_accelerate_install(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--install_accelerate", "--debug"],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; pip install accelerate -U; echo "hello world"; echo "this is a second command" --worker all',
        )

    def test_accelerate_install_version(self):
        output = run_command(
            self.cmd
            + [
                "--config_file",
                "tests/test_configs/latest.yaml",
                "--install_accelerate",
                "--accelerate_version",
                "12.0.0",
                "--debug",
            ],
            return_stdout=True,
        )
        self.assertEqual(
            self.clean_output(output),
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; pip install accelerate==12.0.0; echo "hello world"; echo "this is a second command" --worker all',
        )
