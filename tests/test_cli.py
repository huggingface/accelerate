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

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

import accelerate.commands.test as accelerate_test_cmd
from accelerate.commands.config.config_args import BaseConfig, ClusterConfig, SageMakerConfig, load_config_from_file
from accelerate.commands.estimate import estimate_command, estimate_command_parser, gather_data
from accelerate.commands.launch import _validate_launch_command, launch_command, launch_command_parser
from accelerate.commands.to_fsdp2 import to_fsdp2_command, to_fsdp2_command_parser
from accelerate.commands.tpu import tpu_command_launcher, tpu_command_parser
from accelerate.test_utils.testing import (
    capture_call_output,
    path_in_accelerate_package,
    require_multi_device,
    require_non_hpu,
    require_timm,
    require_transformers,
    run_command,
    run_first,
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
    parser = launch_command_parser()

    @classmethod
    def setUpClass(cls):
        if cls.config_path.is_file():
            cls.config_path.rename(cls.changed_path)

    @classmethod
    def tearDownClass(cls):
        if cls.changed_path.is_file():
            cls.changed_path.rename(cls.config_path)

    @run_first
    def test_no_config(self):
        args = ["--monitor_interval", "0.1", str(self.test_file_path)]
        if torch.cuda.is_available() and (torch.cuda.device_count() > 1):
            args = ["--multi_gpu"] + args
        args = self.parser.parse_args(["--monitor_interval", "0.1", str(self.test_file_path)])
        launch_command(args)

    @run_first
    def test_config_compatibility(self):
        invalid_configs = ["fp8", "invalid", "mpi", "sagemaker"]
        for config in sorted(self.test_config_path.glob("**/*.yaml")):
            if any(invalid_config in str(config) for invalid_config in invalid_configs):
                continue
            with self.subTest(config_file=config):
                args = self.parser.parse_args(["--config_file", str(config), str(self.test_file_path)])
                launch_command(args)

    @run_first
    def test_invalid_keys(self):
        config_path = self.test_config_path / "invalid_keys.yaml"
        with self.assertRaises(
            ValueError,
            msg="The config file at 'invalid_keys.yaml' had unknown keys ('another_invalid_key', 'invalid_key')",
        ):
            args = self.parser.parse_args(["--config_file", str(config_path), str(self.test_file_path)])
            launch_command(args)

    @run_first
    def test_accelerate_test(self):
        args = accelerate_test_cmd.test_command_parser().parse_args([])
        accelerate_test_cmd.test_command(args)

    @run_first
    @require_non_hpu
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

    def test_validate_launch_command(self):
        """Test that the validation function combines args and defaults."""
        parser = launch_command_parser()
        args = parser.parse_args(
            [
                "--num-processes",
                "2",
                "--deepspeed_config_file",
                "path/to/be/accepted",
                "--config-file",
                str(self.test_config_path / "validate_launch_cmd.yaml"),
                "test.py",
            ]
        )
        self.assertFalse(args.debug)
        self.assertTrue(args.fsdp_sync_module_states)
        _validate_launch_command(args)
        self.assertTrue(args.debug)
        self.assertEqual(2, args.num_processes)
        self.assertFalse(args.fsdp_sync_module_states)
        self.assertEqual("path/to/be/accepted", args.deepspeed_config_file)


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
                bad_arg = f"--{arg.replace('_', '-')}"
                # Need an exception for `num-processes` since it's in the docstring
                if bad_arg == "--num-processes":
                    assert help_return.count(bad_arg) == 1, f"Found {bad_arg} in `accelerate launch -h`"
                else:
                    assert bad_arg not in help_return, f"Found {bad_arg} in `accelerate launch -h`"


class ClusterConfigTester(unittest.TestCase):
    """
    Test case for verifying the config dataclasses work
    """

    test_config_path = Path("tests/test_configs")

    def test_base_config(self):
        # Tests that all the dataclasses can be initialized
        config = BaseConfig(
            compute_environment="LOCAL_MACHINE",
            distributed_type="NO",
            mixed_precision="fp16",
            debug=False,
            use_cpu=False,
        )

        assert config.compute_environment == "LOCAL_MACHINE"
        assert config.distributed_type == "NO"
        assert config.mixed_precision == "fp16"
        assert config.debug is False

    def test_cluster_config(self):
        # First normally
        config = ClusterConfig(
            compute_environment="LOCAL_MACHINE",
            distributed_type="NO",
            mixed_precision="fp16",
            num_processes=2,
            debug=False,
            use_cpu=False,
        )

        assert config.compute_environment == "LOCAL_MACHINE"
        assert config.distributed_type == "NO"
        assert config.mixed_precision == "fp16"
        assert config.debug is False

        # Then check with other compute environments
        config = ClusterConfig(
            compute_environment="LOCAL_MACHINE",
            distributed_type="MULTI_GPU",
            mixed_precision="fp16",
            debug=False,
            num_processes=2,
            enable_cpu_affinity=True,
            use_cpu=False,
        )

        assert config.distributed_type == "MULTI_GPU"
        assert config.num_processes == 2
        assert config.enable_cpu_affinity is True

    def test_sagemaker_config(self):
        config = SageMakerConfig(
            compute_environment="AMAZON_SAGEMAKER",
            distributed_type="NO",
            mixed_precision="fp16",
            debug=False,
            use_cpu=False,
            ec2_instance_type="MY_TYPE",
            iam_role_name="MY_ROLE",
        )

        assert config.compute_environment == "AMAZON_SAGEMAKER"
        assert config.ec2_instance_type == "MY_TYPE"
        assert config.iam_role_name == "MY_ROLE"

        config = load_config_from_file(str(self.test_config_path / "0_30_0_sagemaker.yaml"))


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

    def setUp(self):
        self.parser = tpu_command_parser()

    def test_base(self):
        args = self.parser.parse_args(
            ["--command", self.command, "--tpu_zone", self.tpu_zone, "--tpu_name", self.tpu_name, "--debug"]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

    def test_base_backward_compatibility(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                "tests/test_configs/0_12_0.yaml",
                "--command",
                self.command,
                "--tpu_zone",
                self.tpu_zone,
                "--tpu_name",
                self.tpu_name,
                "--debug",
            ]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

    def test_with_config_file(self):
        args = self.parser.parse_args(["--config_file", "tests/test_configs/latest.yaml", "--debug"])
        output = capture_call_output(tpu_command_launcher, args)
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_with_config_file_and_command(self):
        args = self.parser.parse_args(
            ["--config_file", "tests/test_configs/latest.yaml", "--command", self.command, "--debug"]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert f"{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls --worker all" in output

    def test_with_config_file_and_multiple_command(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                "tests/test_configs/latest.yaml",
                "--command",
                self.command,
                "--command",
                'echo "Hello World"',
                "--debug",
            ]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; ls; echo "Hello World" --worker all'
            in output
        )

    def test_with_config_file_and_command_file(self):
        args = self.parser.parse_args(
            ["--config_file", "tests/test_configs/latest.yaml", "--command_file", self.command_file, "--debug"]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_with_config_file_and_command_file_backward_compatibility(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                "tests/test_configs/0_12_0.yaml",
                "--command_file",
                self.command_file,
                "--tpu_zone",
                self.tpu_zone,
                "--tpu_name",
                self.tpu_name,
                "--debug",
            ]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_accelerate_install(self):
        args = self.parser.parse_args(
            ["--config_file", "tests/test_configs/latest.yaml", "--install_accelerate", "--debug"]
        )
        output = capture_call_output(tpu_command_launcher, args)
        assert (
            f'{self.gcloud} test-tpu --zone us-central1-a --command {self.base_output}; pip install accelerate -U; echo "hello world"; echo "this is a second command" --worker all'
            in output
        )

    def test_accelerate_install_version(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                "tests/test_configs/latest.yaml",
                "--install_accelerate",
                "--accelerate_version",
                "12.0.0",
                "--debug",
            ]
        )
        output = capture_call_output(tpu_command_launcher, args)
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
        with self.assertRaises(
            (GatedRepoError, EnvironmentError),
            msg="Repo for model `meta-llama/Llama-2-7b-hf` is gated or environment error occurred",
        ):
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
        largest_layer, total_size = 90669056, 433249280
        # Check that full precision -> int4 is calculating correctly
        assert len(output) == 2, f"Output was missing a precision, expected 2 but received {len(output)}"

        for i, factor in enumerate([1, 2]):
            precision = 32 // factor
            precision_str = f"float{precision}"
            largest_layer_estimate = largest_layer / factor
            total_size_estimate = total_size / factor
            total_training_size_estimate = total_size_estimate * 4

            assert precision_str == output[i][0], f"Output is missing precision `{precision_str}`"
            assert largest_layer_estimate == output[i][1], (
                f"Calculation for largest layer size in `{precision_str}` is incorrect."
            )

            assert total_size_estimate == output[i][2], (
                f"Calculation for total size in `{precision_str}` is incorrect."
            )
            assert total_training_size_estimate == max(output[i][3].values()), (
                f"Calculation for total training size in `{precision_str}` is incorrect."
            )

    @require_transformers
    def test_transformers_model(self):
        args = self.parser.parse_args(["bert-base-cased", "--dtypes", "float32"])
        output = gather_data(args)
        # The largest layer and total size of the model in bytes
        largest_layer, total_size = 90669056, 433249280
        assert largest_layer == output[0][1], (
            f"Calculation for largest layer size in `fp32` is incorrect, expected {largest_layer} but received {output[0][1]}"
        )
        assert total_size == output[0][2], (
            f"Calculation for total size in `fp32` is incorrect, expected {total_size} but received {output[0][2]}"
        )

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
        assert largest_layer == output[0][1], (
            f"Calculation for largest layer size in `fp32` is incorrect, expected {largest_layer} but received {output[0][1]}"
        )
        assert total_size == output[0][2], (
            f"Calculation for total size in `fp32` is incorrect, expected {total_size} but received {output[0][2]}"
        )


class ToFSDP2Tester(unittest.TestCase):
    """
    Test case for verifying the `accelerate to-fsdp2` CLI outputs.
    """

    parser = to_fsdp2_command_parser()
    test_config_path = Path("tests/test_configs")

    @classmethod
    def setUpClass(cls):
        if (cls.test_config_path / "latest_fsdp.yaml").exists():
            cls.original_config = load_config_from_file(str(cls.test_config_path / "latest_fsdp.yaml"))

    @classmethod
    def tearDownClass(cls):
        if cls.original_config is not None:
            cls.original_config.to_yaml_file(str(cls.test_config_path / "latest_fsdp.yaml"))

    def tearDown(self):
        if (self.test_config_path / "output.yaml").exists():
            (self.test_config_path / "output.yaml").unlink()

    def test_nonexistent_config_file(self):
        with self.assertRaises(FileNotFoundError, msg="Config file `nonexistent.yaml` not found"):
            args = self.parser.parse_args(["--config_file", "nonexistent.yaml"])
            to_fsdp2_command(args)

    def test_no_output_without_overwrite(self):
        with self.assertRaises(ValueError, msg="If --overwrite is not set, --output_file must be provided"):
            args = self.parser.parse_args(["--config_file", str(self.test_config_path / "latest_fsdp.yaml")])
            to_fsdp2_command(args)

    @patch("pathlib.Path.exists")
    def test_overwrite_when_output_file_exists(self, mock_exists):
        mock_exists.side_effect = (
            lambda: str(mock_exists._mock_self) == "output.yaml" or mock_exists._mock_self.exists()
        )

        with self.assertRaises(
            FileExistsError, msg="Output file `output.yaml` already exists and --overwrite is not set"
        ):
            args = self.parser.parse_args(
                ["--config_file", str(self.test_config_path / "latest_fsdp.yaml"), "--output_file", "output.yaml"]
            )
            to_fsdp2_command(args)

    def test_fsdp2_config(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                str(self.test_config_path / "latest_fsdp.yaml"),
                "--output_file",
                str(self.test_config_path / "output.yaml"),
            ]
        )
        to_fsdp2_command(args)

        config = load_config_from_file(str(self.test_config_path / "output.yaml"))
        assert isinstance(config, ClusterConfig)
        assert config.fsdp_config["fsdp_version"] == 2

    def test_config_already_fsdp2(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                str(self.test_config_path / "latest_fsdp.yaml"),
                "--output_file",
                str(self.test_config_path / "output.yaml"),
            ]
        )

        mock_config = {"fsdp_config": {"fsdp_version": 2}}

        with patch("accelerate.commands.to_fsdp2.load_config", return_value=mock_config):
            with self.assertLogs(level="WARNING") as cm:
                to_fsdp2_command(args)

            assert "Config already specfies FSDP2, skipping conversion..." in cm.output[0]

    # Has to be the last test because it overwrites the config file
    def test_fsdp2_overwrite(self):
        args = self.parser.parse_args(
            [
                "--config_file",
                str(self.test_config_path / "latest_fsdp.yaml"),
                "--overwrite",
            ]
        )
        to_fsdp2_command(args)

        config = load_config_from_file(str(self.test_config_path / "latest_fsdp.yaml"))
        assert isinstance(config, ClusterConfig)
        assert config.fsdp_config["fsdp_version"] == 2
