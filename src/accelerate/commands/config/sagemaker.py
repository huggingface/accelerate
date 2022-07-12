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

from ...utils.constants import SAGEMAKER_PARALLEL_EC2_INSTANCES
from ...utils.dataclasses import ComputeEnvironment, SageMakerDistributedType
from ...utils.imports import is_boto3_available
from .config_args import SageMakerConfig
from .config_utils import _ask_field, _convert_sagemaker_distributed_mode, _convert_yes_no_to_bool


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
    credentials_configuration = _ask_field(
        "How do you want to authorize? ([0] AWS Profile, [1] Credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)): ",
        lambda x: int(x),
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

    role_management = _ask_field(
        "Do you already have an IAM Role for executing Amazon SageMaker Training Jobs? ([0] provide IAM Role name, [1] create new IAM role using credentials: ",
        lambda x: int(x),
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

    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] data parallelism): ",
        _convert_sagemaker_distributed_mode,
        error_message="Please enter 0 or 1",
    )

    ec2_instance_query = "Which EC2 instance type you want to use for your training "
    if distributed_type != SageMakerDistributedType.NO:
        ec2_instance_query += "("
        for i, instance_type in enumerate(SAGEMAKER_PARALLEL_EC2_INSTANCES):
            ec2_instance_query += f"[{i}] {instance_type}, "
        ec2_instance_query = ec2_instance_query[:-2] + ")? [0]: "
        ec2_instance_type = _ask_field(ec2_instance_query, lambda x: SAGEMAKER_PARALLEL_EC2_INSTANCES[int(x)])
    else:
        ec2_instance_query += "? [ml.p3.2xlarge]:"
        ec2_instance_type = _ask_field(ec2_instance_query, lambda x: str(x).lower(), default="ml.p3.2xlarge")

    num_machines = 1
    if (
        distributed_type == SageMakerDistributedType.DATA_PARALLEL
        or distributed_type == SageMakerDistributedType.MODEL_PARALLEL
    ):
        num_machines = _ask_field(
            "How many machines do you want use? [1]: ",
            lambda x: int(x),
            default=1,
        )

    mixed_precision = _ask_field(
        "Do you wish to use FP16 or BF16 (mixed precision)? [No/FP16/BF16]: ",
        lambda x: str(x),
        default="No",
    )

    return SageMakerConfig(
        image_uri=docker_image,
        compute_environment=ComputeEnvironment.AMAZON_SAGEMAKER,
        distributed_type=distributed_type,
        use_cpu=False,
        ec2_instance_type=ec2_instance_type,
        profile=aws_profile,
        region=aws_region,
        iam_role_name=iam_role_name,
        mixed_precision=mixed_precision,
        num_machines=num_machines,
        sagemaker_inputs_file=sagemaker_inputs_file,
        sagemaker_metrics_file=sagemaker_metrics_file,
    )
