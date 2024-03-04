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
import unittest
from dataclasses import dataclass

import pytest

from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.utils import ComputeEnvironment
from accelerate.utils.launch import _convert_nargs_to_dict


@dataclass
class MockLaunchConfig(SageMakerConfig):
    compute_environment = ComputeEnvironment.AMAZON_SAGEMAKER
    fp16 = True
    ec2_instance_type = "ml.p3.2xlarge"
    iam_role_name = "accelerate_sagemaker_execution_role"
    profile = "hf-sm"
    region = "us-east-1"
    num_machines = 1
    base_job_name = "accelerate-sagemaker-1"
    pytorch_version = "1.6"
    transformers_version = "4.4"
    training_script = "train.py"
    success_training_script_args = [
        "--model_name_or_path",
        "bert",
        "--do_train",
        "False",
        "--epochs",
        "3",
        "--learning_rate",
        "5e-5",
        "--max_steps",
        "50.5",
    ]
    fail_training_script_args = [
        "--model_name_or_path",
        "bert",
        "--do_train",
        "--do_test",
        "False",
        "--do_predict",
        "--epochs",
        "3",
        "--learning_rate",
        "5e-5",
        "--max_steps",
        "50.5",
    ]


class SageMakerLaunch(unittest.TestCase):
    def test_args_convert(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        converted_args = _convert_nargs_to_dict(MockLaunchConfig.success_training_script_args)
        assert isinstance(converted_args["model_name_or_path"], str)
        assert isinstance(converted_args["do_train"], bool)
        assert isinstance(converted_args["epochs"], int)
        assert isinstance(converted_args["learning_rate"], float)
        assert isinstance(converted_args["max_steps"], float)

        with pytest.raises(ValueError):
            _convert_nargs_to_dict(MockLaunchConfig.fail_training_script_args)
