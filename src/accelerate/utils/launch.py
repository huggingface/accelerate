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

import os
import sys

import torch

from ..utils import is_torch_version
from .dataclasses import DistributedType


def get_launch_prefix():
    """
    Grabs the correct launcher for starting a distributed command, such as either `torchrun`, `python -m
    torch.distributed.run`, etc
    """
    if is_torch_version(">=", "1.10.0"):
        cmd = ["torchrun"]
    elif is_torch_version(">=", "1.9.0"):
        cmd = [sys.executable, "-m", "torch.distributed.run"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.launch", "--use_env"]
    return cmd


def _filter_args(args):
    """
    Filters out all `accelerate` specific args
    """
    if is_torch_version(">=", "1.9.1"):
        import torch.distributed.run as distrib_run
    distrib_args = distrib_run.get_args_parser()
    new_args, _ = distrib_args.parse_known_args()

    for key, value in vars(args).items():
        if key in vars(new_args).keys():
            setattr(new_args, key, value)
    return new_args


def env_var_path_add(env_var_name, path_to_add):
    """
    Extends a path-based environment variable's value with a new path and returns the updated value. It's up to the
    caller to set it in os.environ.
    """
    paths = [p for p in os.environ.get(env_var_name, "").split(":") if len(p) > 0]
    paths.append(str(path_to_add))
    return ":".join(paths)


class PrepareForLaunch:
    """
    Prepare a function that will launched in a distributed setup.

    Args:
        launcher (`Callable`):
            The function to launch.
        distributed_type ([`~state.DistributedType`]):
            The distributed type to prepare for.
        debug (`bool`, *optional*, defaults to `False`):
            Whether or not this is a debug launch.
    """

    def __init__(self, launcher, distributed_type="NO", debug=False):
        self.launcher = launcher
        self.distributed_type = DistributedType(distributed_type)
        self.debug = debug

    def __call__(self, index, *args):
        if self.debug:
            world_size = int(os.environ.get("WORLD_SIZE"))
            rdv_file = os.environ.get("ACCELERATE_DEBUG_RDV_FILE")
            torch.distributed.init_process_group(
                "gloo",
                rank=index,
                store=torch.distributed.FileStore(rdv_file, world_size),
                world_size=world_size,
            )
        elif self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_CPU):
            # Prepare the environment for torch.distributed
            os.environ["LOCAL_RANK"] = str(index)
            os.environ["RANK"] = str(index)

        os.environ["FORK_LAUNCHED"] = str(1)
        self.launcher(*args)
