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

from accelerate.utils import write_basic_config

from .config_args import default_json_config_file


description = "Create a default config file for Accelerate with only a few flags set."


def default_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("default-config", description=description)
    else:
        parser = argparse.ArgumentParser("Accelerate default-config command", description=description)

    parser.add_argument(
        "--config_file",
        default=default_json_config_file,
        help=(
            "The path to use to store the config file. Will default to a file named default_config.yaml in the cache "
            "location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
        dest="save_location",
    )

    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        type=str,
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
        default="no",
    )

    if subparsers is not None:
        parser.set_defaults(func=config_command)
    return parser


def config_command(args):
    args = vars(args)
    args.pop("func", None)
    write_basic_config(**args)


def main():
    parser = default_command_parser()
    args = parser.parse_args()
    config_command(args)


if __name__ == "__main__":
    main()
