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
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import accelerate
from accelerate import debug_launcher
from accelerate.commands.launch import _validate_launch_command, launch_command_parser
from accelerate.test_utils import require_cpu, test_ops, test_script
from accelerate.utils.launch import prepare_simple_launcher_cmd_env


@require_cpu
class MultiCPUTester(unittest.TestCase):
    mod_file = inspect.getfile(accelerate.test_utils)
    test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["scripts", "test_cli.py"])
    test_config_path = Path("tests/test_configs")

    def test_cpu(self):
        debug_launcher(test_script.main)

    def test_ops(self):
        debug_launcher(test_ops.main)

    def test_mpi_multicpu_config_cmd(self):
        """
        Parses args a launch command with a test file and the mpi_multicpu.yaml config. Tests getting the command and
        environment vars and verifies the mpirun command arg values.
        """
        mpi_config_path = str(self.test_config_path / "mpi_multicpu.yaml")
        test_file_arg = "--cpu"
        sys.argv = ["accelerate", self.test_file_path, test_file_arg]

        parser = launch_command_parser()
        args = parser.parse_args()
        args.config_file = mpi_config_path
        args, _, _ = _validate_launch_command(args)

        # Mock out the check for mpirun version to simulate Intel MPI
        with patch("shutil.which", return_value=True):
            with patch("subprocess.check_output", return_value=b"Intel MPI"):
                cmd, current_env = prepare_simple_launcher_cmd_env(args)

        # Verify the mpirun command args
        expected_mpirun_cmd = ["mpirun", "-f", "/home/user/hostfile", "-ppn", "4", "-n", "16"]
        self.assertGreater(len(cmd), len(expected_mpirun_cmd))
        generated_mpirun_cmd = cmd[0 : len(expected_mpirun_cmd)]
        self.assertEqual(expected_mpirun_cmd, generated_mpirun_cmd)

        # Verify that the python script and arg
        python_script_cmd = cmd[len(expected_mpirun_cmd) :]
        self.assertEqual(len(python_script_cmd), 3)
        self.assertEqual(python_script_cmd[1], self.test_file_path)
        self.assertEqual(python_script_cmd[2], test_file_arg)
