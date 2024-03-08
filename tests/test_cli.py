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

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from accelerate.commands.estimate import estimate_command, estimate_command_parser, gather_data
from accelerate.commands.launch import _validate_launch_command, launch_command_parser
from accelerate.test_utils import execute_subprocess_async
from accelerate.test_utils.testing import (
    DEFAULT_LAUNCH_COMMAND,
    get_launch_command,
    path_in_accelerate_package,
    require_multi_device,
    require_timm,
    require_transformers,
    run_command,
)
from accelerate.utils import patch_environment
from accelerate.utils.launch import prepare_simple_launcher_cmd_env


class AccelerateLauncherTester(unittest.TestCase):
    """
    Test case for verifying the `accelerate launch` CLI operates correctly.
    If a `default_config.yaml` file is located in the cache it will temporarily move it
    for the duration of the tests.
    """

    test_file_path = path_in_accelerate_package("test_utils", "scripts", "test_cli.py")
    notebook_launcher_path = path_in_accelerate_package("test_utils", "scripts", "test_notebook.py")

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
        if torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            cmd = get_launch_command(multi_gpu=True)
        else:
            cmd = DEFAULT_LAUNCH_COMMAND
        cmd.append(self.test_file_path)
        execute_subprocess_async(cmd, env=os.environ.copy())

    def test_config_compatibility(self):
        for config in sorted(self.test_config_path.glob("**/*.yaml")):
            if "invalid" in str(config) or "mpi" in str(config):
                continue
            with self.subTest(config_file=config):
                cmd = get_launch_command(config_file=config) + [self.test_file_path]
                execute_subprocess_async(cmd)

    def test_invalid_keys(self):
        config_path = self.test_config_path / "invalid_keys.yaml"
        with self.assertRaises(
            RuntimeError,
            msg="The config file at 'invalid_keys.yaml' had unknown keys ('another_invalid_key', 'invalid_key')",
        ):
            cmd = get_launch_command(config_file=config_path) + [self.test_file_path]
            execute_subprocess_async(cmd)

    def test_accelerate_test(self):
        execute_subprocess_async(["accelerate", "test"])

    @require_multi_device
    def test_notebook_launcher(self):
        """
        This test checks a variety of situations and scenarios
        with the `notebook_launcher`
        """
        cmd = ["python", self.notebook_launcher_path]
        with patch_environment(omp_num_threads=1, accelerate_num_processes=2):
            run_command(cmd)

    def test_mpi_multicpu_config_cmd(self):
        """
        Parses a launch command with a test file and the 0_28_0_mpi.yaml config. Tests getting the command and
        environment vars and verifies the mpirun command arg values.
        """
        mpi_config_path = str(self.test_config_path / "0_28_0_mpi.yaml")
        test_file_arg = "--cpu"

        with patch("sys.argv", ["accelerate", str(self.test_file_path), test_file_arg]):
            parser = launch_command_parser()
            args = parser.parse_args()
        args.config_file = mpi_config_path
        args, _, _ = _validate_launch_command(args)

        # Mock out the check for mpirun version to simulate Intel MPI
        with patch("accelerate.utils.launch.which", return_value=True):
            with patch("accelerate.utils.launch.subprocess.check_output", return_value=b"Intel MPI"):
                cmd, _ = prepare_simple_launcher_cmd_env(args)

        # Verify the mpirun command args
        expected_mpirun_cmd = ["mpirun", "-f", "/home/user/hostfile", "-ppn", "4", "-n", "16"]
        self.assertGreater(len(cmd), len(expected_mpirun_cmd))
        generated_mpirun_cmd = cmd[0 : len(expected_mpirun_cmd)]
        self.assertEqual(expected_mpirun_cmd, generated_mpirun_cmd)

        # Verify the python script and args in the mpirun command
        python_script_cmd = cmd[len(expected_mpirun_cmd) :]
        self.assertEqual(len(python_script_cmd), 3)
        self.assertEqual(python_script_cmd[1], str(self.test_file_path))
        self.assertEqual(python_script_cmd[2], test_file_arg)


class LaunchArgTester(unittest.TestCase):
    """
    Test cases revolving around the CLI wrappers
    """

    parser = launch_command_parser()

    def test_hyphen(self):
        # Try a little from each cluster
        args = ["--config-file", "test.yaml", "test.py"]
        result = self.parser.parse_args(args)
        assert result.config_file == "test.yaml"
        assert result.multi_gpu is False

        args = ["--multi-gpu", "--num-processes", "4", "test.py"]
        result = self.parser.parse_args(args)
        assert result.multi_gpu is True
        assert result.num_processes == 4
        # And use a mix
        args = ["--multi-gpu", "--use-deepspeed", "--use-fsdp", "--num_processes", "4", "test.py"]
        result = self.parser.parse_args(args)
        assert result.multi_gpu is True
        assert result.use_deepspeed is True
        assert result.use_fsdp is True
        assert result.num_processes == 4

    def test_underscore(self):
        # Try a little from each cluster
        args = ["--config_file", "test.yaml", "test.py"]
        result = self.parser.parse_args(args)
        assert result.config_file == "test.yaml"

        args = ["--multi_gpu", "--num_processes", "4", "test.py"]
        result = self.parser.parse_args(args)
        assert result.multi_gpu is True
        assert result.num_processes == 4
        # And use a mix
        args = ["--multi_gpu", "--use_deepspeed", "--use_fsdp", "--num-processes", "4", "test.py"]
        result = self.parser.parse_args(args)
        assert result.multi_gpu is True
        assert result.use_deepspeed is True
        assert result.use_fsdp is True
        assert result.num_processes == 4

    def test_duplicate_entities(self):
        help_return = self.parser.format_help()
        args = self.parser.parse_args(["test.py"])
        for arg in args.__dict__:
            if "_" in arg:
                bad_arg = f'--{arg.replace("_", "-")}'
                # Need an exception for `num-processes` since it's in the docstring
                if bad_arg == "--num-processes":
                    assert help_return.count(bad_arg) == 1, f"Found {bad_arg} in `accelerate launch -h`"
                else:
                    assert bad_arg not in help_return, f"Found {bad_arg} in `accelerate launch -h`"


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

    def test_base(self):
        output = run_command(
            self.cmd
            + ["--command", self.command, "--tpu_zone", self.tpu_zone, "--tpu_name", self.tpu_name, "--debug"],
            return_stdout=True,
        )
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

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
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

    def test_with_config_file(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--debug"], return_stdout=True
        )
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_with_config_file_and_command(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--command", self.command, "--debug"],
            return_stdout=True,
        )
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

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
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls; echo "Hello World" --worker all'
            in output
        )

    def test_with_config_file_and_command_file(self):
        output = run_command(
            self.cmd
            + ["--config_file", "tests/test_configs/latest.yaml", "--command_file", self.command_file, "--debug"],
            return_stdout=True,
        )
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
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
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_accelerate_install(self):
        output = run_command(
            self.cmd + ["--config_file", "tests/test_configs/latest.yaml", "--install_accelerate", "--debug"],
            return_stdout=True,
        )
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; pip install accelerate -U; echo "hello world"; echo "this is a second command" --worker all'
            in output
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
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; pip install accelerate==12.0.0; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )


class ModelEstimatorTester(unittest.TestCase):
    """
    Test case for checking the output of `accelerate estimate-memory` is correct.

    - Uses `estimate_command` when trying to catch raised errors
    - Uses `gather_data` when just verifying the calculations are correct
    """

    parser = estimate_command_parser()

    def test_invalid_model_name(self):
        with self.assertRaises(
            RepositoryNotFoundError, msg="Repo for model `somebrokenname` does not exist on the Hub"
        ):
            args = self.parser.parse_args(["somebrokenname"])
            estimate_command(args)

    @require_timm
    def test_invalid_model_name_timm(self):
        with self.assertRaises(RuntimeError, msg="Tried to load `muellerzr/dummy` with `timm` but"):
            args = self.parser.parse_args(["muellerzr/dummy", "--library_name", "timm"])
            estimate_command(args)

    @require_transformers
    def test_invalid_model_name_transformers(self):
        with self.assertRaises(RuntimeError, msg="Tried to load `muellerzr/dummy` with `transformers` but"):
            args = self.parser.parse_args(["muellerzr/dummy", "--library_name", "transformers"])
            estimate_command(args)

    def test_no_metadata(self):
        with self.assertRaises(
            ValueError, msg="Model `muellerzr/dummy` does not have any library metadata on the Hub"
        ):
            args = self.parser.parse_args(["muellerzr/dummy"])
            estimate_command(args)

    def test_gated(self):
        with self.assertRaises(GatedRepoError, msg="Repo for model `meta-llama/Llama-2-7b-hf` is gated"):
            args = self.parser.parse_args(["meta-llama/Llama-2-7b-hf"])
            with patch_environment(hf_hub_disable_implicit_token="1"):
                estimate_command(args)

    @require_transformers
    def test_remote_code(self):
        # Also tests that custom `Auto` classes work
        args = self.parser.parse_args(["hf-internal-testing/test_dynamic_model"])
        with self.assertRaises(ValueError, msg="--trust_remote_code"):
            gather_data(args)

        # Verify it works with the flag
        args = self.parser.parse_args(["hf-internal-testing/test_dynamic_model", "--trust_remote_code"])
        gather_data(args)

    @require_transformers
    def test_explicit_dtypes(self):
        args = self.parser.parse_args(["bert-base-cased", "--dtypes", "float32", "float16"])
        output = gather_data(args)
        # The largest layer and total size of the model in bytes
        largest_layer, total_size = 89075712, 433249280
        # Check that full precision -> int4 is calculating correctly
        assert len(output) == 2, f"Output was missing a precision, expected 2 but received {len(output)}"

        for i, factor in enumerate([1, 2]):
            precision = 32 // factor
            precision_str = f"float{precision}"
            largest_layer_estimate = largest_layer / factor
            total_size_estimate = total_size / factor
            total_training_size_estimate = total_size_estimate * 4

            assert precision_str == output[i][0], f"Output is missing precision `{precision_str}`"
            assert (
                largest_layer_estimate == output[i][1]
            ), f"Calculation for largest layer size in `{precision_str}` is incorrect."

            assert (
                total_size_estimate == output[i][2]
            ), f"Calculation for total size in `{precision_str}` is incorrect."
            assert (
                total_training_size_estimate == output[i][3]
            ), f"Calculation for total training size in `{precision_str}` is incorrect."

    @require_transformers
    def test_transformers_model(self):
        args = self.parser.parse_args(["bert-base-cased", "--dtypes", "float32"])
        output = gather_data(args)
        # The largest layer and total size of the model in bytes
        largest_layer, total_size = 89075712, 433249280
        assert (
            largest_layer == output[0][1]
        ), f"Calculation for largest layer size in `fp32` is incorrect, expected {largest_layer} but received {output[0][1]}"
        assert (
            total_size == output[0][2]
        ), f"Calculation for total size in `fp32` is incorrect, expected {total_size} but received {output[0][2]}"

    @require_transformers
    def test_no_split_modules(self):
        # idefics-80b-instruct has ["IdeficsDecoderLayer", "IdeficsGatedCrossAttentionLayer"]
        args = self.parser.parse_args(["HuggingFaceM4/idefics-80b-instruct", "--dtypes", "float32"])
        output = gather_data(args)
        # without factoring in `no_split` modules, the largest layer is 721420288 bytes
        assert output[0][1] != 721420288, "Largest layer calculation incorrect, did not factor in `no_split` modules."
        # the real answer is 3240165632 bytes
        assert output[0][1] == 3240165632

    @require_timm
    def test_timm_model(self):
        args = self.parser.parse_args(["timm/resnet50.a1_in1k", "--library_name", "timm"])
        output = gather_data(args)
        # The largest layer and total size of the model in bytes
        largest_layer, total_size = 9437184, 102441032
        assert (
            largest_layer == output[0][1]
        ), f"Calculation for largest layer size in `fp32` is incorrect, expected {largest_layer} but received {output[0][1]}"
        assert (
            total_size == output[0][2]
        ), f"Calculation for total size in `fp32` is incorrect, expected {total_size} but received {output[0][2]}"
