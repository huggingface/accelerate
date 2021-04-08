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

from accelerate.state import ComputeEnvironment, SageMakerDistributedType
from accelerate.utils import is_boto3_available

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

    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] data parallelism, [2] model parallelism): ",
        _convert_sagemaker_distributed_mode,
        error_message="Please enter 0, 1 or 2",
    )

    # using the best two instances for single-gpu training or multi-gpu -> can turn into question to make it more diverse
    ec2_instance_type = "ml.p3.2xlarge" if distributed_type == SageMakerDistributedType.NO else "ml.p3dn.24xlarge"
    num_machines = 1
    if (
        distributed_type == SageMakerDistributedType.DATA_PARALLEL
        or distributed_type == SageMakerDistributedType.MODEL_PARALLEL
    ):
        raise NotImplementedError("Model or Data Parallelism is not implemented yet. We are working on it")
        num_machines = _ask_field(
            "How many machines do you want use? [2]: ",
            lambda x: int(x),
            default=2,
        )
    fp16 = _ask_field(
        "Do you wish to use FP16 (mixed precision)? [yes/NO]: ",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )

    return SageMakerConfig(
        compute_environment=ComputeEnvironment.AMAZON_SAGEMAKER,
        distributed_type=distributed_type,
        ec2_instance_type=ec2_instance_type,
        profile=aws_profile,
        region=aws_region,
        iam_role_name=iam_role_name,
        fp16=fp16,
        num_machines=num_machines,
    )
