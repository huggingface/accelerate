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

from accelerate.commands.config import get_config_parser
from accelerate.commands.env import env_command_parser
from accelerate.commands.estimate import estimate_command_parser
from accelerate.commands.launch import launch_command_parser
from accelerate.commands.test import test_command_parser
from accelerate.commands.tpu import tpu_command_parser
from accelerate.commands.utils import CustomArgumentParser


def main():
    parser = CustomArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]", allow_abbrev=False)
    subparsers = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    get_config_parser(subparsers=subparsers)
    estimate_command_parser(subparsers=subparsers)
    env_command_parser(subparsers=subparsers)
    launch_command_parser(subparsers=subparsers)
    tpu_command_parser(subparsers=subparsers)
    test_command_parser(subparsers=subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()
