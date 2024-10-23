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
import sys
from ast import literal_eval
from shutil import which
from typing import Any, Dict, List, Tuple

import torch

from ..commands.config.config_args import SageMakerConfig
from ..utils import (
    DynamoBackend,
    PrecisionType,
    is_fp8_available,
    is_ipex_available,
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_torch_xla_available,
    is_xpu_available,
)
from ..utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS
from ..utils.other import is_port_in_use, merge_dicts
from .dataclasses import DistributedType, SageMakerDistributedType


def _filter_args(args, parser, default_args=[]):
    """
    Filters out all `accelerate` specific args
    """
    new_args, _ = parser.parse_known_args(default_args)
    for key, value in vars(args).items():
        if key in vars(new_args).keys():
            setattr(new_args, key, value)
    return new_args


def _get_mpirun_args():
    """
    Determines the executable and argument names for mpirun, based on the type of install. The supported MPI programs
    are: OpenMPI, Intel MPI, or MVAPICH.

    Returns: Program name and arg names for hostfile, num processes, and processes per node
    """
    # Find the MPI program name
    mpi_apps = [x for x in ["mpirun", "mpiexec"] if which(x)]

    if len(mpi_apps) == 0:
        raise OSError("mpirun or mpiexec were not found. Ensure that Intel MPI, Open MPI, or MVAPICH are installed.")

    # Call the app with the --version flag to determine which MPI app is installed
    mpi_app = mpi_apps[0]
    mpirun_version = subprocess.check_output([mpi_app, "--version"])

    if b"Open MPI" in mpirun_version:
        return mpi_app, "--hostfile", "-n", "--npernode", "--bind-to"
    else:
        # Intel MPI and MVAPICH both use the same arg names
        return mpi_app, "-f", "-n", "-ppn", ""


def setup_fp8_env(args: argparse.Namespace, current_env: Dict[str, str]):
    """
    Setup the FP8 environment variables.
    """
    prefix = "ACCELERATE_"
    for arg in vars(args):
        if arg.startswith("fp8_"):
            value = getattr(args, arg)
            if value is not None:
                current_env[f"{prefix}{arg.upper()}"] = str(getattr(args, arg))
    return current_env


def prepare_simple_launcher_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct simple launcher environment variables.
    """
    cmd = []
    if args.no_python and args.module:
        raise ValueError("--module and --no_python cannot be used together")

    if args.mpirun_hostfile is not None:
        mpi_app_name, hostfile_arg, num_proc_arg, proc_per_node_arg, bind_to_arg = _get_mpirun_args()
        mpirun_ccl = getattr(args, "mpirun_ccl", None)
        bind_to = getattr(args, "bind-to", "socket")
        num_machines = args.num_machines
        num_processes = getattr(args, "num_processes", None)
        nproc_per_node = str(num_processes // num_machines) if num_processes and num_machines else "1"
        cmd += [
            mpi_app_name,
            hostfile_arg,
            args.mpirun_hostfile,
            proc_per_node_arg,
            nproc_per_node,
        ]
        if num_processes:
            cmd += [num_proc_arg, str(num_processes)]
        if bind_to_arg:
            cmd += [bind_to_arg, bind_to]
    if not args.no_python:
        cmd.append(sys.executable)
        if args.module:
            cmd.append("-m")
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    current_env = os.environ.copy()
    current_env["ACCELERATE_USE_CPU"] = str(args.cpu or args.use_cpu)
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
    if args.gpu_ids != "all" and args.gpu_ids is not None:
        if is_xpu_available():
            current_env["ZE_AFFINITY_MASK"] = args.gpu_ids
        elif is_mlu_available():
            current_env["MLU_VISIBLE_DEVICES"] = args.gpu_ids
        elif is_musa_available():
            current_env["MUSA_VISIBLE_DEVICES"] = args.gpu_ids
        elif is_npu_available():
            current_env["ASCEND_RT_VISIBLE_DEVICES"] = args.gpu_ids
        else:
            current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.num_machines > 1:
        current_env["MASTER_ADDR"] = args.main_process_ip
        current_env["MASTER_PORT"] = str(args.main_process_port)

        if args.mpirun_hostfile is not None:
            current_env["CCL_WORKER_COUNT"] = str(mpirun_ccl)
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
    if args.mixed_precision.lower() == "fp8":
        if not is_fp8_available():
            raise RuntimeError(
                "FP8 is not available on this machine. Please ensure that either Transformer Engine or MSAMP is installed."
            )
        current_env = setup_fp8_env(args, current_env)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(
            f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}."
        )
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value
    current_env["ACCELERATE_DYNAMO_MODE"] = args.dynamo_mode
    current_env["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = str(args.dynamo_use_fullgraph)
    current_env["ACCELERATE_DYNAMO_USE_DYNAMIC"] = str(args.dynamo_use_dynamic)

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    if is_ipex_available():
        current_env["ACCELERATE_USE_IPEX"] = str(args.ipex).lower()
        current_env["ACCELERATE_USE_XPU"] = str(args.use_xpu).lower()
    if args.enable_cpu_affinity:
        current_env["ACCELERATE_CPU_AFFINITY"] = "1"
    return cmd, current_env


def prepare_multi_gpu_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    Prepares and returns an environment with the correct multi-GPU environment variables.
    """
    num_processes = args.num_processes
    num_machines = args.num_machines
    main_process_ip = args.main_process_ip
    main_process_port = args.main_process_port
    if num_machines > 1:
        args.nproc_per_node = str(num_processes // num_machines)
        args.nnodes = str(num_machines)
        args.node_rank = int(args.machine_rank)
        if getattr(args, "same_network", False):
            args.master_addr = str(main_process_ip)
            args.master_port = str(main_process_port)
        else:
            args.rdzv_endpoint = f"{main_process_ip}:{main_process_port}"
    else:
        args.nproc_per_node = str(num_processes)
        if main_process_port is not None:
            args.master_port = str(main_process_port)

    if main_process_port is None:
        main_process_port = 29500

    # only need to check port availability in main process, in case we have to start multiple launchers on the same machine
    # for some reasons like splitting log files.
    need_port_check = num_machines <= 1 or int(args.machine_rank) == 0
    if need_port_check and is_port_in_use(main_process_port):
        raise ConnectionError(
            f"Tried to launch distributed communication on port `{main_process_port}`, but another process is utilizing it. "
            "Please specify a different port (such as using the `--main_process_port` flag or specifying a different `main_process_port` in your config file)"
            " and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`."
        )

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        args.module = True
    elif args.no_python:
        args.no_python = True

    current_env = os.environ.copy()
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
    gpu_ids = getattr(args, "gpu_ids", "all")
    if gpu_ids != "all" and args.gpu_ids is not None:
        if is_xpu_available():
            current_env["ZE_AFFINITY_MASK"] = gpu_ids
        elif is_mlu_available():
            current_env["MLU_VISIBLE_DEVICES"] = gpu_ids
        elif is_musa_available():
            current_env["MUSA_VISIBLE_DEVICES"] = gpu_ids
        elif is_npu_available():
            current_env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
        else:
            current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}.")

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)
    if args.mixed_precision.lower() == "fp8":
        if not is_fp8_available():
            raise RuntimeError(
                "FP8 is not available on this machine. Please ensure that either Transformer Engine or MSAMP is installed."
            )
        current_env = setup_fp8_env(args, current_env)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(
            f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}."
        )
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value
    current_env["ACCELERATE_DYNAMO_MODE"] = args.dynamo_mode
    current_env["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = str(args.dynamo_use_fullgraph)
    current_env["ACCELERATE_DYNAMO_USE_DYNAMIC"] = str(args.dynamo_use_dynamic)

    if args.use_fsdp:
        current_env["ACCELERATE_USE_FSDP"] = "true"
        if args.fsdp_cpu_ram_efficient_loading and not args.fsdp_sync_module_states:
            raise ValueError("When using `--fsdp_cpu_ram_efficient_loading` set `--fsdp_sync_module_states` to `True`")

        current_env["FSDP_SHARDING_STRATEGY"] = str(args.fsdp_sharding_strategy)
        current_env["FSDP_OFFLOAD_PARAMS"] = str(args.fsdp_offload_params).lower()
        current_env["FSDP_MIN_NUM_PARAMS"] = str(args.fsdp_min_num_params)
        if args.fsdp_auto_wrap_policy is not None:
            current_env["FSDP_AUTO_WRAP_POLICY"] = str(args.fsdp_auto_wrap_policy)
        if args.fsdp_transformer_layer_cls_to_wrap is not None:
            current_env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = str(args.fsdp_transformer_layer_cls_to_wrap)
        if args.fsdp_backward_prefetch is not None:
            current_env["FSDP_BACKWARD_PREFETCH"] = str(args.fsdp_backward_prefetch)
        if args.fsdp_state_dict_type is not None:
            current_env["FSDP_STATE_DICT_TYPE"] = str(args.fsdp_state_dict_type)
        current_env["FSDP_FORWARD_PREFETCH"] = str(args.fsdp_forward_prefetch).lower()
        current_env["FSDP_USE_ORIG_PARAMS"] = str(args.fsdp_use_orig_params).lower()
        current_env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = str(args.fsdp_cpu_ram_efficient_loading).lower()
        current_env["FSDP_SYNC_MODULE_STATES"] = str(args.fsdp_sync_module_states).lower()
        current_env["FSDP_ACTIVATION_CHECKPOINTING"] = str(args.fsdp_activation_checkpointing).lower()

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
    if args.enable_cpu_affinity:
        current_env["ACCELERATE_CPU_AFFINITY"] = "1"
    return current_env


def prepare_deepspeed_cmd_env(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    """
    Prepares and returns the command list and an environment with the correct DeepSpeed environment variables.
    """
    num_processes = args.num_processes
    num_machines = args.num_machines
    main_process_ip = args.main_process_ip
    main_process_port = args.main_process_port
    cmd = None

    # make sure launcher is not None
    if args.deepspeed_multinode_launcher is None:
        # set to default pdsh
        args.deepspeed_multinode_launcher = DEEPSPEED_MULTINODE_LAUNCHERS[0]

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
        if main_process_ip:
            cmd.extend(["--master_addr", str(main_process_ip)])
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
        args.nproc_per_node = str(num_processes // num_machines)
        args.nnodes = str(num_machines)
        args.node_rank = int(args.machine_rank)
        if getattr(args, "same_network", False):
            args.master_addr = str(main_process_ip)
            args.master_port = str(main_process_port)
        else:
            args.rdzv_endpoint = f"{main_process_ip}:{main_process_port}"
    else:
        args.nproc_per_node = str(num_processes)
        if main_process_port is not None:
            args.master_port = str(main_process_port)

    if main_process_port is None:
        main_process_port = 29500

    # only need to check port availability in main process, in case we have to start multiple launchers on the same machine
    # for some reasons like splitting log files.
    need_port_check = num_machines <= 1 or int(args.machine_rank) == 0
    if need_port_check and is_port_in_use(main_process_port):
        raise ConnectionError(
            f"Tried to launch distributed communication on port `{main_process_port}`, but another process is utilizing it. "
            "Please specify a different port (such as using the `--main_process_port` flag or specifying a different `main_process_port` in your config file)"
            " and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`."
        )

    if args.module and args.no_python:
        raise ValueError("--module and --no_python cannot be used together")
    elif args.module:
        args.module = True
    elif args.no_python:
        args.no_python = True

    current_env = os.environ.copy()
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
    gpu_ids = getattr(args, "gpu_ids", "all")
    if gpu_ids != "all" and args.gpu_ids is not None:
        if is_xpu_available():
            current_env["ZE_AFFINITY_MASK"] = gpu_ids
        elif is_mlu_available():
            current_env["MLU_VISIBLE_DEVICES"] = gpu_ids
        elif is_musa_available():
            current_env["MUSA_VISIBLE_DEVICES"] = gpu_ids
        elif is_npu_available():
            current_env["ASCEND_RT_VISIBLE_DEVICES"] = gpu_ids
        else:
            current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    current_env["PYTHONPATH"] = env_var_path_add("PYTHONPATH", os.path.abspath("."))
    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)
    if args.mixed_precision.lower() == "fp8":
        if not is_fp8_available():
            raise RuntimeError(
                "FP8 is not available on this machine. Please ensure that either Transformer Engine or MSAMP is installed."
            )
        current_env = setup_fp8_env(args, current_env)
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
    if args.enable_cpu_affinity:
        current_env["ACCELERATE_CPU_AFFINITY"] = "1"
    if args.deepspeed_moe_layer_cls_names is not None:
        current_env["ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES"] = str(args.deepspeed_moe_layer_cls_names)
    return cmd, current_env


def prepare_tpu(
    args: argparse.Namespace, current_env: Dict[str, str], pod: bool = False
) -> Tuple[argparse.Namespace, Dict[str, str]]:
    """
    Prepares and returns an environment with the correct TPU environment variables.
    """
    if args.mixed_precision == "bf16" and is_torch_xla_available(check_is_tpu=True):
        if args.downcast_bf16:
            current_env["XLA_DOWNCAST_BF16"] = "1"
        else:
            current_env["XLA_USE_BF16"] = "1"
    if args.debug:
        current_env["ACCELERATE_DEBUG_MODE"] = "true"
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
        raise OSError("You need to provide an aws_access_key_id and aws_secret_access_key when not using aws_profile")

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
        raise ValueError(
            f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DynamoBackend.list()}."
        )

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
    if args.mixed_precision.lower() == "fp8":
        if not is_fp8_available():
            raise RuntimeError(
                "FP8 is not available on this machine. Please ensure that either Transformer Engine or MSAMP is installed."
            )
        environment = setup_fp8_env(args, environment)
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
        elif self.distributed_type in (
            DistributedType.MULTI_GPU,
            DistributedType.MULTI_MLU,
            DistributedType.MULTI_MUSA,
            DistributedType.MULTI_NPU,
            DistributedType.MULTI_XPU,
            DistributedType.MULTI_CPU,
        ):
            # Prepare the environment for torch.distributed
            os.environ["LOCAL_RANK"] = str(index)
            nproc = int(os.environ.get("NPROC", 1))
            node_rank = int(os.environ.get("NODE_RANK", 0))
            os.environ["RANK"] = str(nproc * node_rank + index)

        os.environ["FORK_LAUNCHED"] = str(1)
        self.launcher(*args)
