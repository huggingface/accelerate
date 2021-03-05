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
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from accelerate.commands.config import LaunchConfig, default_config_file
from accelerate.state import DistributedType


class _AddOneArg():
    def __init__(self, launcher):
        self.launcher = launcher
    
    def __call__(self, index):
        self.launcher()


def launch_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("launch")
    else:
        parser = argparse.ArgumentParser("Accelerate launch command")

    parser.add_argument(
        "--config_file", default=None, help="The config file to use for the default values in the launching script."
    )
    parser.add_argument(
        "--multi_gpu",
        default=False,
        action="store_true",
        help="Whether or not this should launch a distributed GPU training.",
    )
    parser.add_argument(
        "--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training."
    )
    parser.add_argument(
        "--fp16", default=False, action="store_true", help="Whether or not to use mixed precision training."
    )
    parser.add_argument(
        "--cpu", default=False, action="store_true", help="Whether or not to force the training on the CPU."
    )
    parser.add_argument(
        "--num_processes", type=int, default=None, help="The total number of processes to be launched in parallel."
    )
    parser.add_argument(
        "--num_machines", type=int, default=1, help="The total number of machines used in this training."
    )
    parser.add_argument(
        "--machine_rank", type=int, default=0, help="The rank of the machine on which this script is launched."
    )
    parser.add_argument(
        "--main_process_ip", type=Optional[str], default=None, help="The IP address of the machine of rank 0."
    )
    parser.add_argument(
        "--main_process_port",
        type=Optional[int],
        default=None,
        help="The port to use to communicate with the machine of rank 0.",
    )
    parser.add_argument(
        "--main_training_function",
        type=str,
        default=None,
        help="The name of the main function to be executed in your script (only for TPU training).",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the script to be launched in parallel, followed by all the arguments for the training "
            "script."
        ),
    )
    # Other arguments of the training scripts
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER, help="Arguments of the training script.")

    if subparsers is not None:
        parser.set_defaults(func=launch_command)
    return parser


def simple_launcher(args):
    cmd = [sys.executable, args.training_script]
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["USE_CPU"] = str(args.cpu)
    current_env["USE_FP16"] = str(args.fp16)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def multi_gpu_launcher(args):
    cmd = [sys.executable, "-m", "torch.distributed.launch"]
    cmd.extend(["--nproc_per_node", str(args.num_processes), "--use_env"])
    if args.num_machines > 1:
        cmd.extend(
            [
                "--nproc_per_node",
                str(args.num_processes // args.num_machines),
                "--nnodes",
                str(args.num_machines),
                "--node_rank",
                str(args.machine_rank),
                "--master_addr",
                args.main_process_ip,
                "--node_rank",
                str(args.main_process_port),
            ]
        )
    else:
        cmd.extend(["--nproc_per_node", str(args.num_processes)])
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["USE_FP16"] = str(args.fp16)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def tpu_launcher(args):
    import torch_xla.distributed.xla_multiprocessing as xmp

    # Import training_script as a module.
    script_path = Path(args.training_script)
    sys.path.append(str(script_path.parent.resolve()))
    mod_name = script_path.stem
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, args.main_training_function):
        raise ValueError(
            f"Your training script should have a function named {args.main_training_function}, or you should pass a "
            "different value to `--main_training_function`."
        )
    main_function = getattr(mod, args.main_training_function)

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args

    # If the function does not take one argument, launch will fail
    launcher_sig = inspect.signature(main_function)
    if len(launcher_sig.parameters) == 0:
        xmp.spawn(_AddOneArg(main_function), args=(), nprocs=args.num_processes)
    else:
        xmp.spawn(main_function, args=(), nprocs=args.num_processes)


def launch_command(args):
    # Sanity checks
    if args.multi_gpu and args.tpu:
        raise ValueError("You can only pick one between `--multi_gpu` and `--tpu`.")

    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file) and not args.cpu:
        defaults = LaunchConfig.from_json_file(json_file=args.config_file)
        if not args.multi_gpu and not args.tpu:
            args.multi_gpu = defaults.distributed_type == DistributedType.MULTI_GPU
            args.tpu = defaults.distributed_type == DistributedType.TPU
        if args.num_processes is None:
            args.num_processes = defaults.num_processes
        if not args.fp16:
            args.fp16 = defaults.fp16
        if args.main_training_function is None:
            args.main_training_function = defaults.main_training_function
    else:
        if args.num_processes is None:
            args.num_processes = 1

    # Use the proper launcher
    if args.multi_gpu and not args.cpu:
        multi_gpu_launcher(args)
    elif args.tpu and not args.cpu:
        tpu_launcher(args)
    else:
        simple_launcher(args)


def main():
    parser = launch_command_parser()
    args = parser.parse_args()
    launch_command(args)


if __name__ == "__main__":
    main()
