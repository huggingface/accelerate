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
import platform
import subprocess

import numpy as np
import psutil
import torch

from accelerate import __version__ as version
from accelerate.commands.config import default_config_file, load_config_from_file

from ..utils import is_mlu_available, is_musa_available, is_npu_available, is_sdaa_available, is_xpu_available


def env_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("env")
    else:
        parser = argparse.ArgumentParser("Accelerate env command")

    parser.add_argument(
        "--config_file", default=None, help="The config file to use for the default values in the launching script."
    )

    if subparsers is not None:
        parser.set_defaults(func=env_command)
    return parser


def env_command(args):
    pt_version = torch.__version__
    pt_cuda_available = torch.cuda.is_available()
    pt_xpu_available = is_xpu_available()
    pt_mlu_available = is_mlu_available()
    pt_sdaa_available = is_sdaa_available()
    pt_musa_available = is_musa_available()
    pt_npu_available = is_npu_available()

    accelerator = "N/A"
    if pt_cuda_available:
        accelerator = "CUDA"
    elif pt_xpu_available:
        accelerator = "XPU"
    elif pt_mlu_available:
        accelerator = "MLU"
    elif pt_sdaa_available:
        accelerator = "SDAA"
    elif pt_musa_available:
        accelerator = "MUSA"
    elif pt_npu_available:
        accelerator = "NPU"

    accelerate_config = "Not found"
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file):
        accelerate_config = load_config_from_file(args.config_file).to_dict()

    # if we can run which, get it
    command = None
    bash_location = "Not found"
    if os.name == "nt":
        command = ["where", "accelerate"]
    elif os.name == "posix":
        command = ["which", "accelerate"]
    if command is not None:
        bash_location = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    info = {
        "`Accelerate` version": version,
        "Platform": platform.platform(),
        "`accelerate` bash location": bash_location,
        "Python version": platform.python_version(),
        "Numpy version": np.__version__,
        "PyTorch version": f"{pt_version}",
        "PyTorch accelerator": accelerator,
        "System RAM": f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
    }
    if pt_cuda_available:
        info["GPU type"] = torch.cuda.get_device_name()
    elif pt_xpu_available:
        info["XPU type"] = torch.xpu.get_device_name()
    elif pt_mlu_available:
        info["MLU type"] = torch.mlu.get_device_name()
    elif pt_sdaa_available:
        info["SDAA type"] = torch.sdaa.get_device_name()
    elif pt_musa_available:
        info["MUSA type"] = torch.musa.get_device_name()
    elif pt_npu_available:
        info["CANN version"] = torch.version.cann

    print("\nCopy-and-paste the text below in your GitHub issue\n")
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]))

    print("- `Accelerate` default config:" if args.config_file is None else "- `Accelerate` config passed:")
    accelerate_config_str = (
        "\n".join([f"\t- {prop}: {val}" for prop, val in accelerate_config.items()])
        if isinstance(accelerate_config, dict)
        else f"\t{accelerate_config}"
    )
    print(accelerate_config_str)

    info["`Accelerate` configs"] = accelerate_config

    return info


def main() -> int:
    parser = env_command_parser()
    args = parser.parse_args()
    env_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
