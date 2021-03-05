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
import json
import os
from dataclasses import dataclass
from typing import Optional

from accelerate.state import DistributedType


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_config_file = os.path.join(cache_dir, "default_config.json")


@dataclass
class LaunchConfig:
    distributed_type: DistributedType
    num_processes: int
    fp16: bool
    machine_rank: int = 0
    num_machines: int = 1
    main_process_ip: Optional[str] = None
    main_process_port: Optional[int] = None
    main_training_function: str = "main"

    @classmethod
    def from_json_file(cls, json_file=None):
        json_file = default_config_file if json_file is None else json_file
        with open(json_file, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_json_file(self, json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            content = json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
            f.write(content)


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


def _ask_field(input_text, convert_value=None, default=None, error_message=None):
    ask_again = True
    while ask_again:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result) if convert_value is not None else result
        except:
            if error_message is not None:
                print(error_message)


def get_user_input():
    def _convert_distributed_mode(value):
        value = int(value)
        return DistributedType(["NO", "MULTI_GPU", "TPU"][value])

    def _convert_yes_no_to_bool(value):
        return {"yes": True, "no": False}[value.lower()]

    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] multi-GPU, [2] TPU): ",
        _convert_distributed_mode,
        error_message="Please enter 0, 1 or 2.",
    )

    machine_rank = 0
    num_machines = 1
    main_process_ip = None
    main_process_port = None
    if distributed_type == DistributedType.MULTI_GPU:
        num_machines = _ask_field(
            "How many different machines will you use (use more than 1 for multi-node training)? [1]: ",
            lambda x: int(x),
            default=1,
        )
        if num_machines > 1:
            machine_rank = _ask_field(
                "What is the rank of this machine (from 0 to the number of machines - 1 )? [0]: ",
                lambda x: int(x),
                default=0,
            )
            main_process_ip = _ask_field(
                "What is the IP address of the machine that will host the main process? ",
            )
            main_process_ip = _ask_field(
                "What is the port you will use to communicate with the main process? ",
                lambda x: int(x),
            )
    if distributed_type == DistributedType.TPU:
        main_training_function = _ask_field(
            "What is the name of the function in your script that should be launched in all parallel scripts? [main]: ",
            default="main",
        )
    else:
        main_training_function = "main"

    num_processes = _ask_field(
        "How many processes in total will you use? [1]: ",
        lambda x: int(x),
        default=1,
        error_message="Please enter an integer.",
    )

    if distributed_type != DistributedType.TPU:
        fp16 = _ask_field(
            "Do you wish to use FP16 (mixed precision)? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
    else:
        fp16 = False

    return LaunchConfig(
        distributed_type=distributed_type,
        num_processes=num_processes,
        fp16=fp16,
        machine_rank=machine_rank,
        num_machines=num_machines,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
        main_training_function=main_training_function,
    )


def config_command(args):
    config = get_user_input()
    if args.config_file is not None:
        config_file = args.config_file
    else:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        config_file = default_config_file

    config.to_json_file(config_file)


def main():
    parser = config_command_parser()
    args = parser.parse_args()
    config_command(args)


if __name__ == "__main__":
    main()
