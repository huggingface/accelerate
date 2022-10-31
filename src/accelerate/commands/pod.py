#!/usr/bin/env python

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import subprocess

from accelerate.commands.config.config_args import default_config_file, load_config_from_file
from packaging.version import Version, parse


_description = "Run commands across a pod of TPU VMs for initial setup before running `accelerate launch`. Will also install Accelerate on the pod."


def pod_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("pod-config", description=_description)
    else:
        parser = argparse.ArgumentParser("Accelerate pod-config command", description=_description)

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the config file to use for accelerate.",
    )

    parser.add_argument(
        "--pod_config_file",
        type=str,
        default=None,
        help="Path to the config file to use for the pod.",
    )

    parser.add_argument(
        "--command_file",
        default=None,
        help="The path to the file containing the commands to run on the pod on startup.",
    )
    parser.add_argument(
        "--command",
        action="append",
        nargs="+",
        help="A command to run on the pod. If not specified, will use the command specified in the command file.",
    )
    parser.add_argument(
        "--tpu_name",
        default=None,
        help="The name of the TPU to use. If not specified, will use the TPU specified in the config file.",
    )
    parser.add_argument(
        "--tpu_zone",
        default=None,
        help="The zone of the TPU to use. If not specified, will use the zone specified in the config file.",
    )
    parser.add_argument(
        "--accelerate_version",
        default="latest",
        help="The version of accelerate to install on the pod. If not specified, will use the latest pypi version. Specify 'dev' to install from GitHub.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="If set, will print the command that would be run instead of running it."
    )

    if subparsers is not None:
        parser.set_defaults(func=pod_launcher)
    return parser


def pod_launcher(args):
    defaults = None

    # Get the default from the config file if it exists.
    if args.config_file is not None or os.path.isfile(default_config_file):
        defaults = load_config_from_file(args.config_file)
        if not args.command_file and defaults.command_file is not None and not args.command:
            args.command_file = defaults.command_file
        if not args.command and defaults.command is not None:
            args.command = defaults.command
        if not args.tpu_name:
            args.tpu_name = defaults.tpu_name
        if not args.tpu_zone:
            args.tpu_zone = defaults.tpu_zone
    if args.accelerate_version == "dev":
        args.accelerate_version = "git+https://github.com/huggingface/accelerate.git"
    elif args.accelerate_version == "latest":
        args.accelerate_version = "accelerate -U"
    elif isinstance(parse(args.accelerate_version), Version):
        args.accelerate_version = f"accelerate=={args.accelerate_version}"

    if not args.command_file and not args.command:
        raise ValueError("You must specify either a command file or a command to run on the pod.")

    if args.command_file:
        with open(args.command_file, "r") as f:
            args.command = [f.read().splitlines()]

    # To turn list of lists into list of strings
    args.command = [line for cmd in args.command for line in cmd]
    # Default to the shared folder and install accelerate
    args.command = ["cd /usr/share", f"pip install {args.accelerate_version}"] + args.command
    args.command = "; ".join(args.command)

    # Then send it to gcloud
    # Eventually try to use google-api-core to do this instead of subprocess
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        args.tpu_name,
        "--zone",
        args.tpu_zone,
        "--command",
        args.command,
        "--worker",
        "all",
    ]
    if args.debug:
        print(cmd)
        return
    subprocess.run(cmd)
    print("Successfully setup pod.")


def main():
    parser = pod_command_parser()
    args = parser.parse_args()

    pod_launcher(args)
