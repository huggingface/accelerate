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

from .config import config_command, config_command_parser
from .config_args import default_config_file, load_config_from_file  # noqa: F401
from .default import default_command_parser, default_config_command


def filter_command_args(args: dict, args_prefix: str):
    "Filters args while only keeping ones that are prefixed with `{args_prefix}.`"
    new_args = argparse.Namespace()
    for key, value in vars(args).items():
        if key.startswith(args_prefix):
            setattr(new_args, key.replace(f"{args_prefix}.", ""), value)
    return new_args


def get_config_parser(subparsers=None):
    parent_parser = argparse.ArgumentParser(add_help=False)
    # The main config parser
    config_parser = config_command_parser(subparsers)

    # Then add other parsers with the parent parser
    default_parser = default_command_parser(config_parser, parents=[parent_parser])  # noqa: F841

    return config_parser


def main():
    config_parser = get_config_parser()
    args = config_parser.parse_args()
    if not args.default:
        args = filter_command_args(args, "config_args")
        config_command(args)
    elif args.default:
        args = filter_command_args(args, "default_args")
        default_config_command(args)


if __name__ == "__main__":
    main()
