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
import logging
import os
import subprocess
import sys
import warnings
from ast import literal_eval
from pathlib import Path
from typing import Dict, List

import torch

import psutil
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.state import get_int_from_env
from accelerate.utils import (
    ComputeEnvironment,
    DistributedType,
    PrecisionType,
    PrepareForLaunch,
    get_launch_prefix,
    is_deepspeed_available,
    is_sagemaker_available,
    patch_environment,
)
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from accelerate.utils.dataclasses import SageMakerDistributedType


logger = logging.getLogger(__name__)


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
        "--use_mps_device",
        default=False,
        action="store_true",
        help="Whether or not this should use MPS-enabled GPU device on MacOS machines.",
    )
    parser.add_argument(
        "--use_deepspeed",
        default=False,
        action="store_true",
        help="Whether to use deepspeed.",
    )
    parser.add_argument(
        "--deepspeed_config_file",
        default=None,
        type=str,
        help="DeepSpeed config file.",
    )
    parser.add_argument(
        "--zero_stage",
        default=None,
        type=int,
        help="DeepSpeed's ZeRO optimization stage (useful only when `use_deepspeed` flag is passed).",
    )
    parser.add_argument(
        "--offload_optimizer_device",
        default=None,
        type=str,
        help="Decides where (none|cpu|nvme) to offload optimizer states (useful only when `use_deepspeed` flag is passed).",
    )
    parser.add_argument(
        "--offload_param_device",
        default=None,
        type=str,
        help="Decides where (none|cpu|nvme) to offload parameters (useful only when `use_deepspeed` flag is passed).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=None,
        type=int,
        help="No of gradient_accumulation_steps used in your training script (useful only when `use_deepspeed` flag is passed).",
    )
    parser.add_argument(
        "--gradient_clipping",
        default=None,
        type=float,
        help="gradient clipping value used in your training script (useful only when `use_deepspeed` flag is passed).",
    )
    parser.add_argument(
        "--zero3_init_flag",
        default=None,
        type=str,
        help="Decides Whether (true|false) to enable `deepspeed.zero.Init` for constructing massive models. "
        "Only applicable with DeepSpeed ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        default=None,
        type=str,
        help="Decides Whether (true|false) to save 16-bit model weights when using ZeRO Stage-3. "
        "Only applicable with DeepSpeed ZeRO Stage-3.",
    )
    parser.add_argument(
        "--deepspeed_hostfile",
        default=None,
        type=str,
        help="DeepSpeed hostfile for configuring multi-node compute resources.",
    )
    parser.add_argument(
        "--deepspeed_exclusion_filter",
        default=None,
        type=str,
        help="DeepSpeed exclusion filter string when using mutli-node setup.",
    )
    parser.add_argument(
        "--deepspeed_inclusion_filter",
        default=None,
        type=str,
        help="DeepSpeed inclusion filter string when using mutli-node setup.",
    )
    parser.add_argument(
        "--deepspeed_multinode_launcher",
        default=None,
        type=str,
        help="DeepSpeed multi-node launcher to use.",
    )
    parser.add_argument(
        "--use_fsdp",
        default=False,
        action="store_true",
        help="Whether to use fsdp.",
    )
    parser.add_argument(
        "--fsdp_offload_params",
        default="false",
        type=str,
        help="Decides Whether (true|false) to offload parameters and gradients to CPU. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=1e8,
        help="FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=int,
        default=1,
        help="FSDP's Sharding Strategy. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_auto_wrap_policy",
        type=str,
        default=None,
        help="FSDP's auto wrap policy. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        default=None,
        type=str,
        help="Transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` .... "
        "(useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_backward_prefetch_policy",
        default=None,
        type=str,
        help="FSDP's backward prefetch policy. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--fsdp_state_dict_type",
        default=None,
        type=str,
        help="FSDP's state dict type. (useful only when `use_fsdp` flag is passed).",
    )
    parser.add_argument(
        "--offload_params",
        default=None,
        type=str,
        help="This argument is deprecated. Use `fsdp_offload_params` instead.",
    )
    parser.add_argument(
        "--min_num_params",
        type=int,
        default=None,
        help="This argument is deprecated. Use `fsdp_min_num_params` instead.",
    )
    parser.add_argument(
        "--sharding_strategy",
        type=int,
        default=None,
        help="This argument is deprecated. Use `fsdp_sharding_strategy` instead.",
    )
    parser.add_argument(
        "--transformer_layer_cls_to_wrap",
        default=None,
        type=str,
        help="This argument is deprecated. Use `fsdp_transformer_layer_cls_to_wrap` instead.",
    )
    parser.add_argument(
        "--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
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
        "--num_machines", type=int, default=None, help="The total number of machines used in this training."
    )
    parser.add_argument(
        "--machine_rank", type=int, default=None, help="The rank of the machine on which this script is launched."
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
        "--downcast_bf16",
        action="store_true",
        help="Whether when using bf16 precision on TPUs if both float and double tensors are cast to bfloat16 or if double tensors remain as float32",
    )
    parser.add_argument(
        "-m",
        "--module",
        action="store_true",
        help="Change each process to interpret the launch script as a Python module, executing with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        action="store_true",
        help="Skip prepending the training script with 'python' - just execute it directly. Useful when the script is not a Python script.",
    )
    parser.add_argument(
        "--num_cpu_threads_per_process",
        type=int,
        default=None,
        help="The number of CPU threads per process. Can be tuned for optimal performance.",
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
    cmd = []
    if args.no_python and args.module:
        raise ValueError("--module and --no_python cannot be used together")
    if not args.no_python:
        cmd.append(sys.executable)
        if args.module:
            cmd.append("-m")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["USE_CPU"] = str(args.cpu or args.use_cpu)
    current_env["USE_MPS_DEVICE"] = str(args.use_mps_device)
    if args.num_machines > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip
        current_env["MASTER_PORT"] = str(args.main_process_port)
    elif args.num_processes > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip if args.main_process_ip is not None else "127.0.0.1"
        current_env["MASTER_PORT"] = str(args.main_process_port) if args.main_process_port is not None else "29500"

    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    current_env["MIXED_PRECISION"] = str(mixed_precision)
    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def multi_gpu_launcher(args):
    cmd = get_launch_prefix()
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
                "--master_port",
                str(args.main_process_port),
            ]
        )
    else:
        cmd.extend(["--nproc_per_node", str(args.num_processes)])
        if args.main_process_port is not None:
            cmd.extend(["--master_port", str(args.main_process_port)])

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        cmd.append("--module")
    elif args.no_python:
        cmd.append("--no_python")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    current_env["MIXED_PRECISION"] = str(mixed_precision)
    if args.use_fsdp:
        if args.sharding_strategy is not None:
            warnings.warn(
                "`sharding_strategy` is deprecated and will be removed in version 0.13.0 of 🤗 Accelerate. Use"
                " `fsdp_sharding_strategy` instead",
                FutureWarning,
            )
            args.fsdp_sharding_strategy = args.sharding_strategy

        if args.offload_params is not None:
            warnings.warn(
                "`offload_params` is deprecated and will be removed in version 0.13.0 of 🤗 Accelerate. Use"
                " `fsdp_offload_params` instead",
                FutureWarning,
            )
            args.fsdp_offload_params = args.offload_params

        if args.min_num_params is not None:
            warnings.warn(
                "`min_num_params` is deprecated and will be removed in version 0.13.0 of 🤗 Accelerate. Use"
                " `fsdp_min_num_params` instead",
                FutureWarning,
            )
            args.fsdp_min_num_params = args.min_num_params

        if args.transformer_layer_cls_to_wrap is not None:
            warnings.warn(
                "`transformer_layer_cls_to_wrap` is deprecated and will be removed in version 0.13.0 of 🤗 Accelerate. Use"
                " `fsdp_transformer_layer_cls_to_wrap` instead",
                FutureWarning,
            )
            args.fsdp_transformer_layer_cls_to_wrap = args.transformer_layer_cls_to_wrap

        current_env["USE_FSDP"] = "true"
        current_env["FSDP_SHARDING_STRATEGY"] = str(args.fsdp_sharding_strategy)
        current_env["FSDP_OFFLOAD_PARAMS"] = str(args.fsdp_offload_params).lower()
        current_env["FSDP_MIN_NUM_PARAMS"] = str(args.fsdp_min_num_params)
        if args.fsdp_auto_wrap_policy is not None:
            current_env["FSDP_AUTO_WRAP_POLICY"] = str(args.fsdp_auto_wrap_policy)
        if args.fsdp_transformer_layer_cls_to_wrap is not None:
            current_env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = str(args.fsdp_transformer_layer_cls_to_wrap)
        if args.fsdp_backward_prefetch_policy is not None:
            current_env["FSDP_BACKWARD_PREFETCH"] = str(args.fsdp_backward_prefetch_policy)
        if args.fsdp_state_dict_type is not None:
            current_env["FSDP_STATE_DICT_TYPE"] = str(args.fsdp_state_dict_type)
    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def deepspeed_launcher(args):
    if not is_deepspeed_available():
        raise ImportError("DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source.")
    cmd = ["deepspeed", "--no_local_rank"]
    if args.num_machines > 1:
        if args.deepspeed_multinode_launcher == DEEPSPEED_MULTINODE_LAUNCHERS[1]:
            cmd = get_launch_prefix()
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
                    "--master_port",
                    str(args.main_process_port),
                ]
            )
        else:
            cmd.extend(
                ["--hostfile", str(args.deepspeed_hostfile), "--launcher", str(args.deepspeed_multinode_launcher)]
            )
            if args.deepspeed_exclusion_filter is not None:
                cmd.extend(
                    [
                        "--exclude",
                        str(args.deepspeed_exclusion_filter),
                    ]
                )
            elif args.deepspeed_inclusion_filter is not None:
                cmd.extend(
                    [
                        "--include",
                        str(args.deepspeed_inclusion_filter),
                    ]
                )
            else:
                cmd.extend(["--num_gpus", str(args.num_processes // args.num_machines)])
    else:
        cmd.extend(["--num_gpus", str(args.num_processes)])

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        cmd.append("--module")
    elif args.no_python:
        cmd.append("--no_python")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    current_env["PYTHONPATH"] = sys.executable
    current_env["MIXED_PRECISION"] = str(mixed_precision)
    current_env["USE_DEEPSPEED"] = "true"
    current_env["DEEPSPEED_ZERO_STAGE"] = str(args.zero_stage)
    current_env["GRADIENT_ACCUMULATION_STEPS"] = str(args.gradient_accumulation_steps)
    current_env["GRADIENT_CLIPPING"] = str(args.gradient_clipping).lower()
    current_env["DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"] = str(args.offload_optimizer_device).lower()
    current_env["DEEPSPEED_OFFLOAD_PARAM_DEVICE"] = str(args.offload_param_device).lower()
    current_env["DEEPSPEED_ZERO3_INIT"] = str(args.zero3_init_flag).lower()
    current_env["DEEPSPEED_ZERO3_SAVE_16BIT_MODEL"] = str(args.zero3_save_16bit_model).lower()
    current_env["DEEPSPEED_CONFIG_FILE"] = str(args.deepspeed_config_file).lower()

    if args.num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        with open(".deepspeed_env", "a") as f:
            for key, value in current_env.items():
                if ";" in value or " " in value:
                    continue
                f.write(f"{key}={value}\n")

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def tpu_launcher(args):
    import torch_xla.distributed.xla_multiprocessing as xmp

    current_env = {}

    if args.no_python:
        raise ValueError("--no_python cannot be used with TPU launcher")

    if args.mixed_precision == "bf16":
        if args.downcast_bf16:
            current_env["XLA_USE_BF16"] = "0"
            current_env["XLA_DOWNCAST_BF16"] = "1"
        else:
            current_env["XLA_USE_BF16"] = "1"
            current_env["XLA_DOWNCAST_BF16"] = "0"

    if args.module:
        mod_name = args.training_script
    else:
        # Import training_script as a module
        script_path = Path(args.training_script)
        sys.path.append(str(script_path.parent.resolve()))
        mod_name = script_path.stem

    mod = importlib.import_module(mod_name)
    if not hasattr(mod, args.main_training_function):
        raise ValueError(
            f"Your training script should have a function named {args.main_training_function}, or you should pass a "
            "different value to `--main_training_function`."
        )

    # Patch sys.argv
    sys.argv = [mod.__file__] + args.training_script_args

    main_function = getattr(mod, args.main_training_function)
    with patch_environment(**current_env):
        xmp.spawn(PrepareForLaunch(main_function), args=(), nprocs=args.num_processes)


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
    if args.module or args.no_python:
        raise ValueError(
            "SageMaker requires a python training script file and cannot be used with --module or --no_python"
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

    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn('--fp16 flag is deprecated. Use "--mixed_precision fp16" instead.', DeprecationWarning)
        mixed_precision = "fp16"

    # Environment variables to be set for use during training job
    environment = {
        "USE_SAGEMAKER": "true",
        "MIXED_PRECISION": str(mixed_precision),
        "SAGEMAKER_DISTRIBUTED_TYPE": sagemaker_config.distributed_type.value,
    }
    # configure distribution set up
    distribution = None
    if sagemaker_config.distributed_type == SageMakerDistributedType.DATA_PARALLEL:
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}

    # configure sagemaker inputs
    sagemaker_inputs = None
    if sagemaker_config.sagemaker_inputs_file is not None:
        print(f"Loading SageMaker Inputs from {sagemaker_config.sagemaker_inputs_file} file")
        sagemaker_inputs = {}
        with open(sagemaker_config.sagemaker_inputs_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split("\t")
                sagemaker_inputs[l[0]] = l[1].strip()
        print(f"Loaded SageMaker Inputs: {sagemaker_inputs}")

    # configure sagemaker metrics
    sagemaker_metrics = None
    if sagemaker_config.sagemaker_metrics_file is not None:
        print(f"Loading SageMaker Metrics from {sagemaker_config.sagemaker_metrics_file} file")
        sagemaker_metrics = []
        with open(sagemaker_config.sagemaker_metrics_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split("\t")
                metric_dict = {
                    "Name": l[0],
                    "Regex": l[1].strip(),
                }
                sagemaker_metrics.append(metric_dict)
        print(f"Loaded SageMaker Metrics: {sagemaker_metrics}")

    # configure session
    print("Creating Estimator")
    huggingface_estimator = HuggingFace(
        image_uri=sagemaker_config.image_uri,
        entry_point=entry_point,
        source_dir=source_dir,
        role=sagemaker_config.iam_role_name,
        transformers_version=sagemaker_config.transformers_version,
        pytorch_version=sagemaker_config.pytorch_version,
        py_version=sagemaker_config.py_version,
        base_job_name=sagemaker_config.base_job_name,
        instance_count=sagemaker_config.num_machines,
        instance_type=sagemaker_config.ec2_instance_type,
        debugger_hook_config=False,
        distribution=distribution,
        hyperparameters=hyperparameters,
        environment=environment,
        metric_definitions=sagemaker_metrics,
    )

    huggingface_estimator.fit(inputs=sagemaker_inputs)
    print(f"You can find your model data at: {huggingface_estimator.model_data}")


def launch_command(args):
    # Sanity checks
    if sum([args.multi_gpu, args.tpu, args.use_deepspeed, args.use_fsdp]) > 1:
        raise ValueError("You can only pick one between `--multi_gpu`, `--use_deepspeed`, `--tpu`, `--use_fsdp`.")

    defaults = None
    warned = []
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file) and not args.cpu:
        defaults = load_config_from_file(args.config_file)
        if (
            not args.multi_gpu
            and not args.tpu
            and not args.use_deepspeed
            and not args.use_fsdp
            and not args.use_mps_device
        ):
            args.use_deepspeed = defaults.distributed_type == DistributedType.DEEPSPEED
            args.multi_gpu = defaults.distributed_type == DistributedType.MULTI_GPU
            args.tpu = defaults.distributed_type == DistributedType.TPU
            args.use_fsdp = defaults.distributed_type == DistributedType.FSDP
            args.use_mps_device = defaults.distributed_type == DistributedType.MPS
        if defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            # Update args with the defaults
            for name, attr in defaults.__dict__.items():
                if isinstance(attr, dict):
                    for k in defaults.deepspeed_config:
                        if getattr(args, k) is None:
                            setattr(args, k, defaults.deepspeed_config[k])
                    for k in defaults.fsdp_config:
                        arg_to_set = k
                        if "fsdp" not in arg_to_set:
                            arg_to_set = "fsdp_" + arg_to_set
                        setattr(args, arg_to_set, defaults.fsdp_config[k])
                    continue

                # Those args are handled separately
                if (
                    name not in ["compute_environment", "fp16", "mixed_precision", "distributed_type"]
                    and getattr(args, name, None) is None
                ):
                    setattr(args, name, attr)
        if not args.mixed_precision:
            if args.fp16:
                args.mixed_precision = "fp16"
            else:
                args.mixed_precision = defaults.mixed_precision
    else:
        if args.num_processes is None:
            warned.append("\t`--num_processes` was set to a value of `1`")
            args.num_processes = 1
        if args.num_machines is None:
            warned.append("\t`--num_machines` was set to a value of `1`")
            args.num_machines = 1
        if args.mixed_precision is None:
            warned.append("\t`--mixed_precision` was set to a value of `'no'`")
            args.mixed_precision = "no"
        if not hasattr(args, "use_cpu"):
            args.use_cpu = args.cpu
    if args.multi_gpu and args.num_processes == 1:
        args.num_processes = torch.cuda.device_count()
        if not any("--num_processes" in warn for warn in warned):
            warned.append(f"\t`--num_processes` was set to `{args.num_processes}`")
        else:
            for i, warn in enumerate(warned):
                if "--num_processes" in warn:
                    warned[i] = warn.replace("`1`", f"`{args.num_processes}`")

    if args.num_cpu_threads_per_process is None:
        local_size = get_int_from_env(
            ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
        )
        args.num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
        if args.num_cpu_threads_per_process == 0:
            args.num_cpu_threads_per_process = 1
        warned.append(
            f"\t`--num_cpu_threads_per_process` was set to `{args.num_cpu_threads_per_process}` to improve out-of-box performance"
        )

    if any(warned):
        message = "The following values were not passed to `accelerate launch` and had defaults used instead:\n"
        message += "\n".join(warned)
        message += (
            "\nTo avoid this warning pass in values for each of the problematic parameters or run `accelerate config`."
        )
        logger.warn(message)

    # Use the proper launcher
    if args.use_deepspeed and not args.cpu:
        deepspeed_launcher(args)
    elif args.use_fsdp and not args.cpu:
        multi_gpu_launcher(args)
    elif args.multi_gpu and not args.cpu:
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
