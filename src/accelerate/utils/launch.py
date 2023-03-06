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
import sys
import warnings
from ast import literal_eval
from typing import Any, Dict, List, Tuple

import torch

from ..commands.config.config_args import SageMakerConfig
from ..commands.config.config_utils import DYNAMO_BACKENDS
from ..utils import DynamoBackend, PrecisionType, is_torch_version
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType


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


def _filter_args(args, parser, default_args=[]):
    """
    Filters out all `accelerate` specific args
    """
    new_args, _ = parser.parse_known_args(default_args)
    for key, value in vars(args).items():
        if key in vars(new_args).keys():
            setattr(new_args, key, value)
    return new_args


def prepare_simple_launcher_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct simple launcher environment variables.
    """
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
        warnings.warn(
            "`mps` is deprecated and will be removed in version 0.18.0 of 🤗 Accelerate."
            " MPS device will be enabled by default when available and can be disabled via `--cpu`.",
            FutureWarning,
        )
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
    current_env["ACCELERATE_DYNAMO_MODE"] = args.dynamo_mode
    current_env["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = str(args.dynamo_use_fullgraph)
    current_env["ACCELERATE_DYNAMO_USE_DYNAMIC"] = str(args.dynamo_use_dynamic)

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    return cmd, current_env


def prepare_multi_gpu_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    Prepares and returns an environment with the correct multi-GPU environment variables.
    """
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
    current_env["ACCELERATE_DYNAMO_MODE"] = args.dynamo_mode
    current_env["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = str(args.dynamo_use_fullgraph)
    current_env["ACCELERATE_DYNAMO_USE_DYNAMIC"] = str(args.dynamo_use_dynamic)

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
    return current_env


def prepare_deepspeed_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct DeepSpeed environment variables.
    """
    num_processes = getattr(args, "num_processes")
    num_machines = getattr(args, "num_machines")
    main_process_ip = getattr(args, "main_process_ip")
    main_process_port = getattr(args, "main_process_port")
    cmd = None

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
    return cmd, current_env


def prepare_tpu(
    args: argparse.Namespace, current_env: Dict[str, str], pod: bool = False
) -> Tuple[argparse.Namespace, Dict[str, str]]:
    """
    Prepares and returns an environment with the correct TPU environment variables.
    """
    current_env["XLA_USE_BF16"] = "0"
    current_env["XLA_DOWNCAST_BF16"] = "0"
    if args.mixed_precision == "bf16":
        if args.downcast_bf16:
            current_env["XLA_DOWNCAST_BF16"] = "1"
        else:
            current_env["XLA_USE_BF16"] = "1"
    if pod:
        # Take explicit args and set them up for XLA
        args.vm = args.tpu_vm
        args.tpu = args.tpu_name
    return args, current_env


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


def prepare_sagemager_args_inputs(
    sagemaker_config: SageMakerConfig, args: argparse.Namespace
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
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
        "ACCELERATE_DYNAMO_MODE": args.dynamo_mode,
        "ACCELERATE_DYNAMO_USE_FULLGRAPH": str(args.dynamo_use_fullgraph),
        "ACCELERATE_DYNAMO_USE_DYNAMIC": str(args.dynamo_use_dynamic),
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
    args = {
        "image_uri": sagemaker_config.image_uri,
        "entry_point": entry_point,
        "source_dir": source_dir,
        "role": sagemaker_config.iam_role_name,
        "transformers_version": sagemaker_config.transformers_version,
        "pytorch_version": sagemaker_config.pytorch_version,
        "py_version": sagemaker_config.py_version,
        "base_job_name": sagemaker_config.base_job_name,
        "instance_count": sagemaker_config.num_machines,
        "instance_type": sagemaker_config.ec2_instance_type,
        "debugger_hook_config": False,
        "distribution": distribution,
        "hyperparameters": hyperparameters,
        "environment": environment,
        "metric_definitions": sagemaker_metrics,
    }

    if sagemaker_config.additional_args is not None:
        args = merge_dicts(sagemaker_config.additional_args, args)
    return args, sagemaker_inputs


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
