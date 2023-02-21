#!/usr/bin/env python

# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import torch.distributed.run as distrib_run


ignored_params = [
    "nproc_per_node",
    "nnodes",
    "nproc_per_node",
    "help",
    "rdzv_endpoint",
    "training_script",
    "training_script_args",
]

changed_name = {
    "node_rank": "machine_rank",
}


def add_arguments(argument_group: argparse._ArgumentGroup):
    distrib_parser = distrib_run.get_args_parser()
    for action in distrib_parser._actions:
        if action.dest in ignored_params:
            continue
        if action.dest in changed_name.keys():
            action.dest = changed_name[action.dest]
            action.option_strings = [f"--{action.dest}"]
        argument_group._add_action(action)
        action.container = argument_group
