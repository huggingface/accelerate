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

from accelerate.test_utils import execute_subprocess_async, path_in_accelerate_package


def test_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("test")
    else:
        parser = argparse.ArgumentParser("Accelerate test command")

    parser.add_argument(
        "--config_file",
        default=None,
        help=(
            "The path to use to store the config file. Will default to a file named default_config.yaml in the cache "
            "location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
    )

    if subparsers is not None:
        parser.set_defaults(func=test_command)
    return parser


def test_command(args):
    script_name = path_in_accelerate_package("test_utils", "scripts", "test_script.py")

    if args.config_file is None:
        test_args = [script_name]
    else:
        test_args = f"--config_file={args.config_file} {script_name}".split()

    cmd = ["accelerate-launch"] + test_args
    result = execute_subprocess_async(cmd)
    if result.returncode == 0:
        print("Test is a success! You are ready for your distributed training!")


def main():
    parser = test_command_parser()
    args = parser.parse_args()
    test_command(args)


if __name__ == "__main__":
    main()
