#!/usr/bin/env python

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
import json
import os

from ...utils.constants import SAGEMAKER_PARALLEL_EC2_INSTANCES, TORCH_DYNAMO_MODES
from ...utils.dataclasses import ComputeEnvironment, SageMakerDistributedType
from ...utils.imports import is_boto3_available
from .config_args import SageMakerConfig
from .config_utils import (
    DYNAMO_BACKENDS,
    _ask_field,
    _ask_options,
    _convert_dynamo_backend,
    _convert_mixed_precision,
    _convert_sagemaker_distributed_mode,
    _convert_yes_no_to_bool,
)


if is_boto3_available():
    import boto3  # noqa: F401


def _create_iam_role_for_sagemaker(role_name):
    iam_client = boto3.client("iam")

    sagemaker_trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }
    try:
        # create the role, associated with the chosen trust policy
        iam_client.create_role(
            RoleName=role_name, AssumeRolePolicyDocument=json.dumps(sagemaker_trust_policy, indent=2)
        )
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:*",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetAuthorizationToken",
                        "cloudwatch:PutMetricData",
                        "cloudwatch:GetMetricData",
                        "cloudwatch:GetMetricStatistics",
                        "cloudwatch:ListMetrics",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                        "logs:GetLogEvents",
                        "s3:CreateBucket",
                        "s3:ListBucket",
                        "s3:GetBucketLocation",
                        "s3:GetObject",
                        "s3:PutObject",
                    ],
                    "Resource": "*",
                }
            ],
        }
        # attach policy to role
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}_policy_permission",
            PolicyDocument=json.dumps(policy_document, indent=2),
        )
    except iam_client.exceptions.EntityAlreadyExistsException:
        print(f"role {role_name} already exists. Using existing one")


def _get_iam_role_arn(role_name):
    iam_client = boto3.client("iam")
    return iam_client.get_role(RoleName=role_name)["Role"]["Arn"]


def get_sagemaker_input():
    credentials_configuration = _ask_options(
        "How do you want to authorize?",
        ["AWS Profile", "Credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) "],
        int,
    )
    aws_profile = None
    if credentials_configuration == 0:
        aws_profile = _ask_field("Enter your AWS Profile name: [default] ", default="default")
        os.environ["AWS_PROFILE"] = aws_profile
    else:
        print(
            "Note you will need to provide AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY when you launch you training script with,"
            "`accelerate launch --aws_access_key_id XXX --aws_secret_access_key YYY`"
        )
        aws_access_key_id = _ask_field("AWS Access Key ID: ")
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id

        aws_secret_access_key = _ask_field("AWS Secret Access Key: ")
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    aws_region = _ask_field("Enter your AWS Region: [us-east-1]", default="us-east-1")
    os.environ["AWS_DEFAULT_REGION"] = aws_region

    role_management = _ask_options(
        "Do you already have an IAM Role for executing Amazon SageMaker Training Jobs?",
        ["Provide IAM Role name", "Create new IAM role using credentials"],
        int,
    )
    if role_management == 0:
        iam_role_name = _ask_field("Enter your IAM role name: ")
    else:
        iam_role_name = "accelerate_sagemaker_execution_role"
        print(f'Accelerate will create an iam role "{iam_role_name}" using the provided credentials')
        _create_iam_role_for_sagemaker(iam_role_name)

    is_custom_docker_image = _ask_field(
        "Do you want to use custom Docker image? [yes/NO]: ",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )
    docker_image = None
    if is_custom_docker_image:
        docker_image = _ask_field("Enter your Docker image: ", lambda x: str(x).lower())

    is_sagemaker_inputs_enabled = _ask_field(
        "Do you want to provide SageMaker input channels with data locations? [yes/NO]: ",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )
    sagemaker_inputs_file = None
    if is_sagemaker_inputs_enabled:
        sagemaker_inputs_file = _ask_field(
            "Enter the path to the SageMaker inputs TSV file with columns (channel_name, data_location): ",
            lambda x: str(x).lower(),
        )

    is_sagemaker_metrics_enabled = _ask_field(
        "Do you want to enable SageMaker metrics? [yes/NO]: ",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )
    sagemaker_metrics_file = None
    if is_sagemaker_metrics_enabled:
        sagemaker_metrics_file = _ask_field(
            "Enter the path to the SageMaker metrics TSV file with columns (metric_name, metric_regex): ",
            lambda x: str(x).lower(),
        )

    distributed_type = _ask_options(
        "What is the distributed mode?",
        ["No distributed training", "Data parallelism"],
        _convert_sagemaker_distributed_mode,
    )
    dynamo_config = {}
    use_dynamo = _ask_field(
        "Do you wish to optimize your script with torch dynamo?[yes/NO]:",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )
    if use_dynamo:
        prefix = "dynamo_"
        dynamo_config[prefix + "backend"] = _ask_options(
            "Which dynamo backend would you like to use?",
            [x.lower() for x in DYNAMO_BACKENDS],
            _convert_dynamo_backend,
            default=2,
        )
        use_custom_options = _ask_field(
            "Do you want to customize the defaults sent to torch.compile? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )

        if use_custom_options:
            dynamo_config[prefix + "mode"] = _ask_options(
                "Which mode do you want to use?",
                TORCH_DYNAMO_MODES,
                lambda x: TORCH_DYNAMO_MODES[int(x)],
                default="default",
            )
            dynamo_config[prefix + "use_fullgraph"] = _ask_field(
                "Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            dynamo_config[prefix + "use_dynamic"] = _ask_field(
                "Do you want to enable dynamic shape tracing? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            dynamo_config[prefix + "use_regional_compilation"] = _ask_field(
                "Do you want to enable regional compilation? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )

    ec2_instance_query = "Which EC2 instance type you want to use for your training?"
    if distributed_type != SageMakerDistributedType.NO:
        ec2_instance_type = _ask_options(
            ec2_instance_query, SAGEMAKER_PARALLEL_EC2_INSTANCES, lambda x: SAGEMAKER_PARALLEL_EC2_INSTANCES[int(x)]
        )
    else:
        ec2_instance_query += "? [ml.p3.2xlarge]:"
        ec2_instance_type = _ask_field(ec2_instance_query, lambda x: str(x).lower(), default="ml.p3.2xlarge")

    debug = False
    if distributed_type != SageMakerDistributedType.NO:
        debug = _ask_field(
            "Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )

    num_machines = 1
    if distributed_type in (SageMakerDistributedType.DATA_PARALLEL, SageMakerDistributedType.MODEL_PARALLEL):
        num_machines = _ask_field(
            "How many machines do you want use? [1]: ",
            int,
            default=1,
        )

    mixed_precision = _ask_options(
        "Do you wish to use FP16 or BF16 (mixed precision)?",
        ["no", "fp16", "bf16", "fp8"],
        _convert_mixed_precision,
    )

    if use_dynamo and mixed_precision == "no":
        print(
            "Torch dynamo used without mixed precision requires TF32 to be efficient. Accelerate will enable it by default when launching your scripts."
        )

    return SageMakerConfig(
        image_uri=docker_image,
        compute_environment=ComputeEnvironment.AMAZON_SAGEMAKER,
        distributed_type=distributed_type,
        use_cpu=False,
        dynamo_config=dynamo_config,
        ec2_instance_type=ec2_instance_type,
        profile=aws_profile,
        region=aws_region,
        iam_role_name=iam_role_name,
        mixed_precision=mixed_precision,
        num_machines=num_machines,
        sagemaker_inputs_file=sagemaker_inputs_file,
        sagemaker_metrics_file=sagemaker_metrics_file,
        debug=debug,
    )
