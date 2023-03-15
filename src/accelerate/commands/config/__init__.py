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

from .config import config_command_parser
from .config_args import default_config_file, load_config_from_file  # noqa: F401
from .default import default_command_parser
from .update import update_command_parser


def get_config_parser(subparsers=None):
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # The main config parser
    config_parser = config_command_parser(subparsers)
    # The subparser to add commands to
    subcommands = config_parser.add_subparsers(title="subcommands", dest="subcommand")

    # Then add other parsers with the parent parser
    default_command_parser(subcommands, parents=[parent_parser])
    update_command_parser(subcommands, parents=[parent_parser])

    return config_parser


def main():
    config_parser = get_config_parser()
    args = config_parser.parse_args()

    if not hasattr(args, "func"):
        config_parser.print_help()
        exit(1)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()
