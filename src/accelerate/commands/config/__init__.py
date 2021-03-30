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

import argparse
import os

from accelerate.state import ComputeEnvironment

from .cluster import get_cluster_input
from .config_args import cache_dir, default_config_file, default_yaml_config_file, load_config_from_file  # noqa: F401
from .config_utils import _ask_field, _convert_compute_environment
from .sagemaker import get_sagemaker_input


def get_user_input():
    compute_environment = _ask_field(
        "In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): ",
        _convert_compute_environment,
        error_message="Please enter 0 or 1",
    )
    if compute_environment == ComputeEnvironment.AMAZON_SAGEMAKER:
        config = get_sagemaker_input()
    else:
        config = get_cluster_input()
    return config


def config_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("config")
    else:
        parser = argparse.ArgumentParser("Accelerate config command")

    parser.add_argument(
        "--config_file",
        default=None,
        help=(
            "The path to use to store the config file. Will default to a file named default_config.json in the cache "
            "location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
    )

    if subparsers is not None:
        parser.set_defaults(func=config_command)
    return parser


def config_command(args):
    config = get_user_input()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        config_file = default_yaml_config_file

    if config_file.endswith(".json"):
        config.to_json_file(config_file)
    else:
        config.to_yaml_file(config_file)


def main():
    parser = config_command_parser()
    args = parser.parse_args()
    config_command(args)


if __name__ == "__main__":
    main()
