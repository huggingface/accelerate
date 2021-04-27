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

from accelerate.test_utils import execute_subprocess_async


def test_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("test")
    else:
        parser = argparse.ArgumentParser("Accelerate test command")

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
        parser.set_defaults(func=test_command)
    return parser


def test_command(args):
    script_name = os.path.sep.join(__file__.split(os.path.sep)[:-2] + ["test_utils", "test_script.py"])

    test_args = f"""
        --config_file={args.config_file} {script_name}
    """.split()
    cmd = ["accelerate-launch"] + test_args
    result = execute_subprocess_async(cmd, env=os.environ.copy())
    if result.returncode == 0:
        print("Test is a success! You are ready for your distributed training!")


def main():
    parser = test_command_parser()
    args = parser.parse_args()
    test_command(args)


if __name__ == "__main__":
    main()
