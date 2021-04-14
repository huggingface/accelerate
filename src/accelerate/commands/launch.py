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
from ast import literal_eval
from pathlib import Path
from typing import Dict, List

from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.state import ComputeEnvironment, DistributedType
from accelerate.utils import is_sagemaker_available


class _AddOneArg:
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
    parser.add_argument("--main_process_ip", type=str, default=None, help="The IP address of the machine of rank 0.")
    parser.add_argument(
        "--main_process_port",
        type=int,
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
        "--aws_access_key_id",
        type=str,
        default=None,
        help="The AWS_ACCESS_KEY_ID used to launch the Amazon SageMaker training job",
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="The AWS_SECRET_ACCESS_KEY used to launch the Amazon SageMaker training job",
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


def _convert_nargs_to_dict(nargs: List[str]) -> Dict[str, str]:
    if len(nargs) < 0:
        return {}
    # helper function to infer type for argsparser

    def _infer_type(s):
        try:
            s = float(s)

            if s // 1 == s:
                return int(s)
            return s
        except ValueError:
            return s

    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args(nargs)
    for index, argument in enumerate(unknown):
        if argument.startswith(("-", "--")):
            action = None
            if index + 1 < len(unknown):  # checks if next index would be in list
                if unknown[index + 1].startswith(("-", "--")):  # checks if next element is an key
                    # raise an error if element is store_true or store_false
                    raise ValueError(
                        "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                    )
            else:  # raise an error if last element is store_true or store_false
                raise ValueError(
                    "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                )
            # adds argument to parser based on action_store true
            if action is None:
                parser.add_argument(argument, type=_infer_type)
            else:
                parser.add_argument(argument, action=action)

    return {
        key: (literal_eval(value) if value == "True" or value == "False" else value)
        for key, value in parser.parse_args(nargs).__dict__.items()
    }


def sagemaker_launcher(sagemaker_config: SageMakerConfig, args):
    if not is_sagemaker_available():
        raise ImportError(
            "Please install sagemaker to be able to launch training on Amazon SageMaker with `pip install accelerate[sagemaker]`"
        )
    from sagemaker.huggingface import HuggingFace

    # configure environment
    print("Configuring Amazon SageMaker environment")
    os.environ["AWS_DEFAULT_REGION"] = sagemaker_config.region

    # configure credentials
    if sagemaker_config.profile is not None:
        os.environ["AWS_PROFILE"] = sagemaker_config.profile
    elif args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key
    else:
        raise EnvironmentError(
            "You need to provide an aws_access_key_id and aws_secret_access_key when not using aws_profile"
        )

    # extract needed arguments
    source_dir = os.path.dirname(args.training_script)
    if not source_dir:  # checks if string is empty
        source_dir = "."
    entry_point = os.path.basename(args.training_script)
    if not entry_point.endswith(".py"):
        raise ValueError(f'Your training script should be a python script and not "{entry_point}"')

    print("Converting Arguments to Hyperparameters")
    hyperparameters = _convert_nargs_to_dict(args.training_script_args)

    environment = {"USE_FP16": args.fp16}  # Environment variables to be set for use during training job

    # configure distribution set up
    distribution = None  # TODO: not yet implemented

    # configure session
    print("Creating Estimator")
    huggingface_estimator = HuggingFace(
        entry_point=entry_point,
        source_dir=source_dir,
        role=sagemaker_config.iam_role_name,
        transformers_version="4.4",
        pytorch_version="1.6",
        py_version="py36",
        base_job_name=sagemaker_config.base_job_name,
        instance_count=sagemaker_config.num_machines,
        instance_type=sagemaker_config.ec2_instance_type,
        debugger_hook_config=False,
        distribution=distribution,
        hyperparameters=hyperparameters,
        environment=environment,
    )

    huggingface_estimator.fit()
    print(f"You can find your model data at: {huggingface_estimator.model_data}")


def launch_command(args):
    # Sanity checks
    if args.multi_gpu and args.tpu:
        raise ValueError("You can only pick one between `--multi_gpu` and `--tpu`.")

    defaults = None
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file) and not args.cpu:
        defaults = load_config_from_file(args.config_file)
        if not args.multi_gpu and not args.tpu:
            args.multi_gpu = defaults.distributed_type == DistributedType.MULTI_GPU
            args.tpu = defaults.distributed_type == DistributedType.TPU
        if args.num_processes is None and defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            args.num_processes = defaults.num_processes
        if not args.fp16:
            args.fp16 = defaults.fp16
        if args.main_training_function is None and defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            args.main_training_function = defaults.main_training_function
    else:
        if args.num_processes is None:
            args.num_processes = 1

    # Use the proper launcher
    if args.multi_gpu and not args.cpu:
        multi_gpu_launcher(args)
    elif args.tpu and not args.cpu:
        tpu_launcher(args)
    elif defaults is not None and defaults.compute_environment == ComputeEnvironment.AMAZON_SAGEMAKER:
        sagemaker_launcher(defaults, args)
    else:
        simple_launcher(args)


def main():
    parser = launch_command_parser()
    args = parser.parse_args()
    launch_command(args)


if __name__ == "__main__":
    main()
