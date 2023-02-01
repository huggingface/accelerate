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
from ast import literal_eval
from pathlib import Path
from typing import Dict, List

import torch

import psutil
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.state import get_int_from_env
from accelerate.utils import (
    ComputeEnvironment,
    DistributedType,
    DynamoBackend,
    PrecisionType,
    PrepareForLaunch,
    _filter_args,
    is_deepspeed_available,
    is_rich_available,
    is_sagemaker_available,
    is_torch_version,
    patch_environment,
)
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from accelerate.utils.dataclasses import SageMakerDistributedType
from accelerate.utils.launch import env_var_path_add


if is_rich_available():
    from rich import get_console
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


logger = logging.getLogger(__name__)

options_to_group = {
    "--multi-gpu": "Distributed GPUs",
    "--tpu": "TPU",
    "--mps": "MPS",
    "--use_deepspeed": "DeepSpeed Arguments",
    "--use_fsdp": "FSDP Arguments",
    "--use_megatron_lm": "Megatron-LM Arguments",
}


def clean_option(option):
    "Finds all cases of - after the first two characters and changes them to _"
    if option.startswith("--"):
        return option[:3] + option[3:].replace("-", "_")


class _CustomHelpAction(argparse._HelpAction):
    """
    This is a custom help action that will hide all arguments that are not used in the command line when the help is
    called. This is useful for the case where the user is using a specific platform and only wants to see the arguments
    for that platform.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if "accelerate" in sys.argv[0] and "launch" in sys.argv[1:]:
            args = sys.argv[2:]
        else:
            args = sys.argv[1:]
        opts = parser._actions
        titles = [
            "Hardware Selection Arguments",
            "Resource Selection Arguments",
            "Training Paradigm Arguments",
            "positional arguments",
            "optional arguments",
        ]
        if len(args) > 1:
            used_platforms = [arg for arg in args if arg in options_to_group.keys()]
            args = list(map(clean_option, args))
            used_titles = [options_to_group[o] for o in used_platforms]
            for i, arg in enumerate(opts):
                # If the argument's container is outside of the used titles, hide it
                if arg.container.title not in titles + used_titles:
                    setattr(opts[i], "help", argparse.SUPPRESS)
                # If the argument is hardware selection, but not being passed, hide it
                elif arg.container.title == "Hardware Selection Arguments":
                    if set(arg.option_strings).isdisjoint(set(args)):
                        setattr(opts[i], "help", argparse.SUPPRESS)
                    else:
                        setattr(opts[i], "help", arg.help + " (currently selected)")
                # If the argument is a training paradigm, but not being passed, hide it
                elif arg.container.title == "Training Paradigm Arguments":
                    if set(arg.option_strings).isdisjoint(set(used_platforms)):
                        setattr(opts[i], "help", argparse.SUPPRESS)
                    else:
                        setattr(opts[i], "help", arg.help + " (currently selected)")
            for i, group in enumerate(list(parser._action_groups)):
                # If all arguments in the group are hidden, hide the group
                if all([arg.help == argparse.SUPPRESS for arg in group._group_actions]):
                    parser._action_groups.remove(group)

        super().__call__(parser, namespace, values, option_string)


def launch_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("launch", add_help=False)
    else:
        parser = argparse.ArgumentParser("Accelerate launch command", add_help=False)

    parser.register("action", "help", _CustomHelpAction)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

    parser.add_argument(
        "--config_file", default=None, help="The config file to use for the default values in the launching script."
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Silence subprocess errors from the launch stack trace and only show the relevant tracebacks. (Only applicable to DeepSpeed and single-process configurations)",
    )
    # Hardware selection arguments
    hardware_args = parser.add_argument_group(
        "Hardware Selection Arguments", "Arguments for selecting the hardware to be used."
    )
    hardware_args.add_argument(
        "--cpu", default=False, action="store_true", help="Whether or not to force the training on the CPU."
    )
    hardware_args.add_argument(
        "--mps",
        default=False,
        action="store_true",
        help="Whether or not this should use MPS-enabled GPU device on MacOS machines.",
    )
    hardware_args.add_argument(
        "--multi_gpu",
        default=False,
        action="store_true",
        help="Whether or not this should launch a distributed GPU training.",
    )
    hardware_args.add_argument(
        "--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training."
    )

    # Resource selection arguments
    resource_args = parser.add_argument_group(
        "Resource Selection Arguments", "Arguments for fine-tuning how available hardware should be used."
    )
    resource_args.add_argument(
        "--dynamo_backend",
        type=str,
        choices=["no"] + [b.lower() for b in DYNAMO_BACKENDS],
        help="Choose a backend to optimize your training with dynamo, see more at "
        "https://github.com/pytorch/torchdynamo.",
    )
    resource_args.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
    )
    resource_args.add_argument(
        "--num_processes", type=int, default=None, help="The total number of processes to be launched in parallel."
    )
    resource_args.add_argument(
        "--num_machines", type=int, default=None, help="The total number of machines used in this training."
    )
    resource_args.add_argument(
        "--num_cpu_threads_per_process",
        type=int,
        default=None,
        help="The number of CPU threads per process. Can be tuned for optimal performance.",
    )

    # Training Paradigm arguments
    paradigm_args = parser.add_argument_group(
        "Training Paradigm Arguments", "Arguments for selecting which training paradigm to be used."
    )
    paradigm_args.add_argument(
        "--use_deepspeed",
        default=False,
        action="store_true",
        help="Whether to use deepspeed.",
    )
    paradigm_args.add_argument(
        "--use_fsdp",
        default=False,
        action="store_true",
        help="Whether to use fsdp.",
    )
    paradigm_args.add_argument(
        "--use_megatron_lm",
        default=False,
        action="store_true",
        help="Whether to use Megatron-LM.",
    )

    # distributed GPU training arguments
    distributed_args = parser.add_argument_group("Distributed GPUs", "Arguments related to distributed GPU training.")
    distributed_args.add_argument(
        "--gpu_ids",
        default=None,
        help="What GPUs (by id) should be used for training on this machine as a comma-seperated list",
    )
    distributed_args.add_argument(
        "--same_network",
        default=False,
        action="store_true",
        help="Whether all machines used for multinode training exist on the same local network.",
    )
    distributed_args.add_argument(
        "--machine_rank", type=int, default=None, help="The rank of the machine on which this script is launched."
    )
    distributed_args.add_argument(
        "--main_process_ip", type=str, default=None, help="The IP address of the machine of rank 0."
    )
    distributed_args.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="The port to use to communicate with the machine of rank 0.",
    )
    # Rendezvous related arguments
    distributed_args.add_argument(
        "--rdzv_conf",
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    distributed_args.add_argument(
        "--max_restarts",
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    distributed_args.add_argument(
        "--monitor_interval",
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
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

    # tpu arguments
    tpu_args = parser.add_argument_group("TPU", "Arguments related to TPU.")
    tpu_args.add_argument(
        "--main_training_function",
        type=str,
        default=None,
        help="The name of the main function to be executed in your script (only for TPU training).",
    )
    tpu_args.add_argument(
        "--downcast_bf16",
        action="store_true",
        help="Whether when using bf16 precision on TPUs if both float and double tensors are cast to bfloat16 or if double tensors remain as float32.",
    )

    # DeepSpeed arguments
    deepspeed_args = parser.add_argument_group("DeepSpeed Arguments", "Arguments related to DeepSpeed.")
    deepspeed_args.add_argument(
        "--deepspeed_config_file",
        default=None,
        type=str,
        help="DeepSpeed config file.",
    )
    deepspeed_args.add_argument(
        "--zero_stage",
        default=None,
        type=int,
        help="DeepSpeed's ZeRO optimization stage (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to `2`.",
    )
    deepspeed_args.add_argument(
        "--offload_optimizer_device",
        default=None,
        type=str,
        help="Decides where (none|cpu|nvme) to offload optimizer states (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to 'none'.",
    )
    deepspeed_args.add_argument(
        "--offload_param_device",
        default=None,
        type=str,
        help="Decides where (none|cpu|nvme) to offload parameters (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to 'none'.",
    )
    deepspeed_args.add_argument(
        "--gradient_accumulation_steps",
        default=None,
        type=int,
        help="No of gradient_accumulation_steps used in your training script (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to `1`.",
    )
    deepspeed_args.add_argument(
        "--gradient_clipping",
        default=None,
        type=float,
        help="gradient clipping value used in your training script (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to `1.0`.",
    )
    deepspeed_args.add_argument(
        "--zero3_init_flag",
        default=None,
        type=str,
        help="Decides Whether (true|false) to enable `deepspeed.zero.Init` for constructing massive models. "
        "Only applicable with DeepSpeed ZeRO Stage-3. If unspecified, will default to `true`.",
    )
    deepspeed_args.add_argument(
        "--zero3_save_16bit_model",
        default=None,
        type=str,
        help="Decides Whether (true|false) to save 16-bit model weights when using ZeRO Stage-3. "
        "Only applicable with DeepSpeed ZeRO Stage-3. If unspecified, will default to `false`.",
    )
    deepspeed_args.add_argument(
        "--deepspeed_hostfile",
        default=None,
        type=str,
        help="DeepSpeed hostfile for configuring multi-node compute resources.",
    )
    deepspeed_args.add_argument(
        "--deepspeed_exclusion_filter",
        default=None,
        type=str,
        help="DeepSpeed exclusion filter string when using mutli-node setup.",
    )
    deepspeed_args.add_argument(
        "--deepspeed_inclusion_filter",
        default=None,
        type=str,
        help="DeepSpeed inclusion filter string when using mutli-node setup.",
    )
    deepspeed_args.add_argument(
        "--deepspeed_multinode_launcher",
        default=None,
        type=str,
        help="DeepSpeed multi-node launcher to use. If unspecified, will default to `pdsh`.",
    )

    # fsdp arguments
    fsdp_args = parser.add_argument_group("FSDP Arguments", "Arguments related to Fully Shared Data Parallelism.")
    fsdp_args.add_argument(
        "--fsdp_offload_params",
        default="false",
        type=str,
        help="Decides Whether (true|false) to offload parameters and gradients to CPU. (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=1e8,
        help="FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_sharding_strategy",
        type=int,
        default=1,
        help="FSDP's Sharding Strategy. (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_auto_wrap_policy",
        type=str,
        default=None,
        help="FSDP's auto wrap policy. (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        default=None,
        type=str,
        help="Transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` .... "
        "(useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_backward_prefetch_policy",
        default=None,
        type=str,
        help="FSDP's backward prefetch policy. (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_state_dict_type",
        default=None,
        type=str,
        help="FSDP's state dict type. (useful only when `use_fsdp` flag is passed).",
    )

    # megatron_lm args
    megatron_lm_args = parser.add_argument_group("Megatron-LM Arguments", "Arguments related to Megatron-LM.")
    megatron_lm_args.add_argument(
        "--megatron_lm_tp_degree",
        type=int,
        default=1,
        help="Megatron-LM's Tensor Parallelism (TP) degree. (useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_pp_degree",
        type=int,
        default=1,
        help="Megatron-LM's Pipeline Parallelism (PP) degree. (useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_num_micro_batches",
        type=int,
        default=None,
        help="Megatron-LM's number of micro batches when PP degree > 1. (useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_sequence_parallelism",
        default=None,
        type=str,
        help="Decides Whether (true|false) to enable Sequence Parallelism when TP degree > 1. "
        "(useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_recompute_activations",
        default=None,
        type=str,
        help="Decides Whether (true|false) to enable Selective Activation Recomputation. "
        "(useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_use_distributed_optimizer",
        default=None,
        type=str,
        help="Decides Whether (true|false) to use distributed optimizer "
        "which shards optimizer state and gradients across Data Pralellel (DP) ranks. "
        "(useful only when `use_megatron_lm` flag is passed).",
    )
    megatron_lm_args.add_argument(
        "--megatron_lm_gradient_clipping",
        default=1.0,
        type=float,
        help="Megatron-LM's gradient clipping value based on global L2 Norm (0 to disable). "
        "(useful only when `use_megatron_lm` flag is passed).",
    )

    # AWS arguments
    aws_args = parser.add_argument_group("AWS Arguments", "Arguments related to AWS.")
    aws_args.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="The AWS_ACCESS_KEY_ID used to launch the Amazon SageMaker training job",
    )
    aws_args.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="The AWS_SECRET_ACCESS_KEY used to launch the Amazon SageMaker training job.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print out the torch.distributed stack trace when something fails.",
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
    current_env["ACCELERATE_USE_CPU"] = str(args.cpu or args.use_cpu)
    current_env["ACCELERATE_USE_MPS_DEVICE"] = str(args.mps)
    if args.mps:
        current_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif args.gpu_ids != "all" and args.gpu_ids is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
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

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DYNAMO_BACKENDS}.")
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        if not args.quiet:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        else:
            sys.exit(1)


def multi_gpu_launcher(args):
    if is_torch_version(">=", "1.9.1"):
        import torch.distributed.run as distrib_run
    else:
        raise NotImplementedError("Native multi-GPU training requires pytorch>=1.9.1")
    num_processes = getattr(args, "num_processes")
    num_machines = getattr(args, "num_machines")
    main_process_ip = getattr(args, "main_process_ip")
    main_process_port = getattr(args, "main_process_port")
    if num_machines > 1:
        setattr(args, "nproc_per_node", str(num_processes // num_machines))
        setattr(args, "nnodes", str(num_machines))
        setattr(args, "node_rank", int(args.machine_rank))
        if getattr(args, "same_network", False):
            setattr(args, "master_addr", str(main_process_ip))
            setattr(args, "master_port", str(main_process_port))
        else:
            setattr(args, "rdzv_endpoint", f"{main_process_ip}:{main_process_port}")
    else:
        setattr(args, "nproc_per_node", str(num_processes))
        if main_process_port is not None:
            setattr(args, "master_port", str(main_process_port))

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        setattr(args, "module", True)
    elif args.no_python:
        setattr(args, "no_python", True)

    current_env = os.environ.copy()
    gpu_ids = getattr(args, "gpu_ids", "all")
    if gpu_ids != "all" and args.gpu_ids is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.")

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DYNAMO_BACKENDS}.")
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value

    if args.use_fsdp:
        current_env["ACCELERATE_USE_FSDP"] = "true"
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

    if args.use_megatron_lm:
        prefix = "MEGATRON_LM_"
        current_env["ACCELERATE_USE_MEGATRON_LM"] = "true"
        current_env[prefix + "TP_DEGREE"] = str(args.megatron_lm_tp_degree)
        current_env[prefix + "PP_DEGREE"] = str(args.megatron_lm_pp_degree)
        current_env[prefix + "GRADIENT_CLIPPING"] = str(args.megatron_lm_gradient_clipping)
        if args.megatron_lm_num_micro_batches is not None:
            current_env[prefix + "NUM_MICRO_BATCHES"] = str(args.megatron_lm_num_micro_batches)
        if args.megatron_lm_sequence_parallelism is not None:
            current_env[prefix + "SEQUENCE_PARALLELISM"] = str(args.megatron_lm_sequence_parallelism)
        if args.megatron_lm_recompute_activations is not None:
            current_env[prefix + "RECOMPUTE_ACTIVATIONS"] = str(args.megatron_lm_recompute_activations)
        if args.megatron_lm_use_distributed_optimizer is not None:
            current_env[prefix + "USE_DISTRIBUTED_OPTIMIZER"] = str(args.megatron_lm_use_distributed_optimizer)

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    debug = getattr(args, "debug", False)
    args = _filter_args(args)
    with patch_environment(**current_env):
        try:
            distrib_run.run(args)
        except:
            if is_rich_available() and debug:
                console = get_console()
                console.print("\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]")
                console.print_exception(suppress=[__file__], show_locals=False)


def deepspeed_launcher(args):
    if is_torch_version(">=", "1.9.1"):
        import torch.distributed.run as distrib_run
    if not is_deepspeed_available():
        raise ImportError("DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source.")
    num_processes = getattr(args, "num_processes")
    num_machines = getattr(args, "num_machines")
    main_process_ip = getattr(args, "main_process_ip")
    main_process_port = getattr(args, "main_process_port")

    # make sure launcher is not None
    if args.deepspeed_multinode_launcher is None:
        # set to default pdsh
        setattr(args, "deepspeed_multinode_launcher", DEEPSPEED_MULTINODE_LAUNCHERS[0])

    if num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        cmd = ["deepspeed", "--no_local_rank"]
        cmd.extend(["--hostfile", str(args.deepspeed_hostfile), "--launcher", str(args.deepspeed_multinode_launcher)])
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
        cmd.extend(["--master_port", str(main_process_port)])
        if args.module and args.no_python:
            raise ValueError("--module and --no_python cannot be used together")
        elif args.module:
            cmd.append("--module")
        elif args.no_python:
            cmd.append("--no_python")
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)
    elif num_machines > 1 and args.deepspeed_multinode_launcher == DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        setattr(args, "nproc_per_node", str(num_processes // num_machines))
        setattr(args, "nnodes", str(num_machines))
        setattr(args, "node_rank", int(args.machine_rank))
        if getattr(args, "same_network", False):
            setattr(args, "master_addr", str(main_process_ip))
            setattr(args, "master_port", str(main_process_port))
        else:
            setattr(args, "rdzv_endpoint", f"{main_process_ip}:{main_process_port}")
    else:
        setattr(args, "nproc_per_node", str(num_processes))
        if main_process_port is not None:
            setattr(args, "master_port", str(main_process_port))

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        setattr(args, "module", True)
    elif args.no_python:
        setattr(args, "no_python", True)

    current_env = os.environ.copy()
    gpu_ids = getattr(args, "gpu_ids", "all")
    if gpu_ids != "all" and args.gpu_ids is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    current_env["PYTHONPATH"] = env_var_path_add("PYTHONPATH", os.path.abspath("."))
    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)
    current_env["ACCELERATE_CONFIG_DS_FIELDS"] = str(args.deepspeed_fields_from_accelerate_config).lower()
    current_env["ACCELERATE_USE_DEEPSPEED"] = "true"
    if args.zero_stage is not None:
        current_env["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = str(args.zero_stage)
    if args.gradient_accumulation_steps is not None:
        current_env["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(args.gradient_accumulation_steps)
    if args.gradient_clipping is not None:
        current_env["ACCELERATE_GRADIENT_CLIPPING"] = str(args.gradient_clipping).lower()
    if args.offload_optimizer_device is not None:
        current_env["ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"] = str(args.offload_optimizer_device).lower()
    if args.offload_param_device is not None:
        current_env["ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE"] = str(args.offload_param_device).lower()
    if args.zero3_init_flag is not None:
        current_env["ACCELERATE_DEEPSPEED_ZERO3_INIT"] = str(args.zero3_init_flag).lower()
    if args.zero3_save_16bit_model is not None:
        current_env["ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL"] = str(args.zero3_save_16bit_model).lower()
    if args.deepspeed_config_file is not None:
        current_env["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = str(args.deepspeed_config_file)

    if args.num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        with open(".deepspeed_env", "a") as f:
            for key, value in current_env.items():
                if ";" in value or " " in value:
                    continue
                f.write(f"{key}={value}\n")

        process = subprocess.Popen(cmd, env=current_env)
        process.wait()
        if process.returncode != 0:
            if not args.quiet:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
            else:
                sys.exit(1)
    else:
        if is_torch_version("<", "1.9.1"):
            raise NotImplementedError("Multi-node training requires pytorch>=1.9.1")

        debug = getattr(args, "debug", False)
        args = _filter_args(args)
        with patch_environment(**current_env):
            try:
                distrib_run.run(args)
            except:
                if is_rich_available() and debug:
                    console = get_console()
                    console.print("\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]")
                    console.print_exception(suppress=[__file__], show_locals=False)


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
        key: (literal_eval(value) if value in ("True", "False") else value)
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

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DYNAMO_BACKENDS}.")

    # Environment variables to be set for use during training job
    environment = {
        "ACCELERATE_USE_SAGEMAKER": "true",
        "ACCELERATE_MIXED_PRECISION": str(mixed_precision),
        "ACCELERATE_DYNAMO_BACKEND": dynamo_backend.value,
        "ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE": sagemaker_config.distributed_type.value,
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
    if sum([args.multi_gpu, args.cpu, args.tpu, args.mps, args.use_deepspeed, args.use_fsdp]) > 1:
        raise ValueError(
            "You can only use one of `--cpu`, `--multi_gpu`, `--mps`, `--tpu`, `--use_deepspeed`, `--use_fsdp` at a time."
        )
    if args.multi_gpu and (args.num_processes is not None) and (args.num_processes < 2):
        raise ValueError("You need to use at least 2 processes to use `--multi_gpu`.")

    defaults = None
    warned = []
    mp_from_config_flag = False
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file) and not args.cpu:
        defaults = load_config_from_file(args.config_file)
        if (
            not args.multi_gpu
            and not args.tpu
            and not args.mps
            and not args.use_deepspeed
            and not args.use_fsdp
            and not args.use_megatron_lm
        ):
            args.use_deepspeed = defaults.distributed_type == DistributedType.DEEPSPEED
            args.multi_gpu = defaults.distributed_type == DistributedType.MULTI_GPU
            args.tpu = defaults.distributed_type == DistributedType.TPU
            args.use_fsdp = defaults.distributed_type == DistributedType.FSDP
            args.mps = defaults.distributed_type == DistributedType.MPS
            args.use_megatron_lm = defaults.distributed_type == DistributedType.MEGATRON_LM
        if not args.mps:
            if args.gpu_ids is None:
                if defaults.gpu_ids is not None:
                    args.gpu_ids = defaults.gpu_ids
                else:
                    args.gpu_ids = "all"
            if len(args.gpu_ids.split(",")) < 2 and args.multi_gpu and (args.gpu_ids != "all"):
                args.multi_gpu = False
        if defaults.compute_environment == ComputeEnvironment.LOCAL_MACHINE:
            # Update args with the defaults
            for name, attr in defaults.__dict__.items():
                if isinstance(attr, dict):
                    for k in defaults.deepspeed_config:
                        setattr(args, k, defaults.deepspeed_config[k])
                    for k in defaults.fsdp_config:
                        arg_to_set = k
                        if "fsdp" not in arg_to_set:
                            arg_to_set = "fsdp_" + arg_to_set
                        setattr(args, arg_to_set, defaults.fsdp_config[k])
                    for k in defaults.megatron_lm_config:
                        setattr(args, k, defaults.megatron_lm_config[k])
                    continue

                # Those args are handled separately
                if (
                    name not in ["compute_environment", "mixed_precision", "distributed_type"]
                    and getattr(args, name, None) is None
                ):
                    setattr(args, name, attr)
        if not args.mixed_precision:
            if defaults.mixed_precision is None:
                args.mixed_precision = "no"
            else:
                args.mixed_precision = defaults.mixed_precision
                mp_from_config_flag = True

        if args.dynamo_backend is None:
            warned.append("\t`--dynamo_backend` was set to a value of `'no'`")
            args.dynamo_backend = "no"
    else:
        if args.num_processes is None:
            args.num_processes = torch.cuda.device_count() if args.multi_gpu else 1
            warned.append(f"\t`--num_processes` was set to a value of `{args.num_processes}`")
        if args.num_machines is None:
            warned.append("\t`--num_machines` was set to a value of `1`")
            args.num_machines = 1
        if args.mixed_precision is None:
            warned.append("\t`--mixed_precision` was set to a value of `'no'`")
            args.mixed_precision = "no"
        if not hasattr(args, "use_cpu"):
            args.use_cpu = args.cpu
        if args.dynamo_backend is None:
            warned.append("\t`--dynamo_backend` was set to a value of `'no'`")
            args.dynamo_backend = "no"

    is_aws_env_disabled = defaults is None or (
        defaults is not None and defaults.compute_environment != ComputeEnvironment.AMAZON_SAGEMAKER
    )
    if is_aws_env_disabled and args.num_cpu_threads_per_process is None:
        args.num_cpu_threads_per_process = 1
        if args.use_cpu and args.num_processes > 1:
            local_size = get_int_from_env(
                ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
            )
            threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
            if threads_per_process > 1:
                args.num_cpu_threads_per_process = threads_per_process
                warned.append(
                    f"\t`--num_cpu_threads_per_process` was set to `{args.num_cpu_threads_per_process}` to improve out-of-box performance when training on CPUs"
                )

    if any(warned):
        message = "The following values were not passed to `accelerate launch` and had defaults used instead:\n"
        message += "\n".join(warned)
        message += (
            "\nTo avoid this warning pass in values for each of the problematic parameters or run `accelerate config`."
        )
        logger.warning(message)

    # Use the proper launcher
    if args.use_deepspeed and not args.cpu:
        args.deepspeed_fields_from_accelerate_config = list(defaults.deepspeed_config.keys()) if defaults else []
        if mp_from_config_flag:
            args.deepspeed_fields_from_accelerate_config.append("mixed_precision")
        args.deepspeed_fields_from_accelerate_config = ",".join(args.deepspeed_fields_from_accelerate_config)
        deepspeed_launcher(args)
    elif args.use_fsdp and not args.cpu:
        multi_gpu_launcher(args)
    elif args.use_megatron_lm and not args.cpu:
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
