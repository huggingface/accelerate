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
from pathlib import Path

import psutil
import torch

from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
    ComputeEnvironment,
    DistributedType,
    PrepareForLaunch,
    _filter_args,
    check_cuda_p2p_ib_support,
    convert_dict_to_env_variables,
    is_bf16_available,
    is_deepspeed_available,
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_rich_available,
    is_sagemaker_available,
    is_torch_version,
    is_torch_xla_available,
    is_xpu_available,
    patch_environment,
    prepare_deepspeed_cmd_env,
    prepare_multi_gpu_env,
    prepare_sagemager_args_inputs,
    prepare_simple_launcher_cmd_env,
    prepare_tpu,
    str_to_bool,
)
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES


if is_rich_available():
    from rich import get_console
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


logger = logging.getLogger(__name__)


options_to_group = {
    "multi_gpu": "Distributed GPUs",
    "tpu": "TPU",
    "use_deepspeed": "DeepSpeed Arguments",
    "use_fsdp": "FSDP Arguments",
    "use_megatron_lm": "Megatron-LM Arguments",
    "fp8_backend": "FP8 Arguments",
}


def clean_option(option):
    "Finds all cases of - after the first two characters and changes them to _"
    if "fp8_backend" in option:
        option = "--fp8_backend"
    if option.startswith("--"):
        return option[2:].replace("-", "_")


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This is a custom help formatter that will hide all arguments that are not used in the command line when the help is
    called. This is useful for the case where the user is using a specific platform and only wants to see the arguments
    for that platform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.titles = [
            "Hardware Selection Arguments",
            "Resource Selection Arguments",
            "Training Paradigm Arguments",
            "positional arguments",
            "optional arguments",
        ]

    def add_argument(self, action: argparse.Action):
        if "accelerate" in sys.argv[0] and "launch" in sys.argv[1:]:
            args = sys.argv[2:]
        else:
            args = sys.argv[1:]

        if len(args) > 1:
            args = list(map(clean_option, args))
            used_platforms = [arg for arg in args if arg in options_to_group.keys()]
            used_titles = [options_to_group[o] for o in used_platforms]
            if action.container.title not in self.titles + used_titles:
                action.help = argparse.SUPPRESS
            elif action.container.title == "Hardware Selection Arguments":
                if set(action.option_strings).isdisjoint(set(args)):
                    action.help = argparse.SUPPRESS
                else:
                    action.help = action.help + " (currently selected)"
            elif action.container.title == "Training Paradigm Arguments":
                if set(action.option_strings).isdisjoint(set(args)):
                    action.help = argparse.SUPPRESS
                else:
                    action.help = action.help + " (currently selected)"

        action.option_strings = [s for s in action.option_strings if "-" not in s[2:]]
        super().add_argument(action)

    def end_section(self):
        if len(self._current_section.items) < 2:
            self._current_section.items = []
            self._current_section.heading = ""
        super().end_section()


def launch_command_parser(subparsers=None):
    description = "Launch a python script in a distributed scenario. Arguments can be passed in with either hyphens (`--num-processes=2`) or underscores (`--num_processes=2`)"
    if subparsers is not None:
        parser = subparsers.add_parser(
            "launch", description=description, add_help=False, allow_abbrev=False, formatter_class=CustomHelpFormatter
        )
    else:
        parser = CustomArgumentParser(
            "Accelerate launch command",
            description=description,
            add_help=False,
            allow_abbrev=False,
            formatter_class=CustomHelpFormatter,
        )

    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")

    parser.add_argument(
        "--config_file",
        default=None,
        help="The config file to use for the default values in the launching script.",
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
        "--multi_gpu",
        default=False,
        action="store_true",
        help="Whether or not this should launch a distributed GPU training.",
    )
    hardware_args.add_argument(
        "--tpu", default=False, action="store_true", help="Whether or not this should launch a TPU training."
    )
    hardware_args.add_argument(
        "--ipex",
        default=False,
        action="store_true",
        help="Whether or not this should launch a Intel PyTorch Extension (IPEX) training.",
    )

    # Resource selection arguments
    resource_args = parser.add_argument_group(
        "Resource Selection Arguments", "Arguments for fine-tuning how available hardware should be used."
    )
    resource_args.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16", "fp8"],
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
    resource_args.add_argument(
        "--enable_cpu_affinity",
        default=False,
        action="store_true",
        help="Whether or not CPU affinity and balancing should be enabled. Currently only supported on NVIDIA hardware.",
    )
    # Dynamo arguments
    resource_args.add_argument(
        "--dynamo_backend",
        type=str,
        choices=["no"] + [b.lower() for b in DYNAMO_BACKENDS],
        help="Choose a backend to optimize your training with dynamo, see more at "
        "https://github.com/pytorch/torchdynamo.",
    )
    resource_args.add_argument(
        "--dynamo_mode",
        type=str,
        default="default",
        choices=TORCH_DYNAMO_MODES,
        help="Choose a mode to optimize your training with dynamo.",
    )
    resource_args.add_argument(
        "--dynamo_use_fullgraph",
        default=False,
        action="store_true",
        help="Whether to use full graph mode for dynamo or it is ok to break model into several subgraphs",
    )
    resource_args.add_argument(
        "--dynamo_use_dynamic",
        default=False,
        action="store_true",
        help="Whether to enable dynamic shape tracing.",
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
    paradigm_args.add_argument(
        "--use_xpu",
        default=False,
        action="store_true",
        help="Whether to use IPEX plugin to speed up training on XPU specifically.",
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
    distributed_args.add_argument(
        "-t",
        "--tee",
        default="0",
        type=str,
        help="Tee std streams into a log file and also to console.",
    )
    distributed_args.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Base directory to use for log files when using torchrun/torch.distributed.run as launcher. "
            "Use with --tee to redirect std streams info log files."
        ),
    )
    distributed_args.add_argument(
        "--role",
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    # Rendezvous related arguments
    distributed_args.add_argument(
        "--rdzv_backend",
        type=str,
        default="static",
        help="The rendezvous method to use, such as 'static' (the default) or 'c10d'",
    )
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
        default=0.1,
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

    # TPU arguments
    tpu_args = parser.add_argument_group("TPU", "Arguments related to TPU.")
    tpu_args.add_argument(
        "--tpu_cluster",
        action="store_true",
        dest="tpu_use_cluster",
        help="Whether to use a GCP TPU pod for training.",
    )
    tpu_args.add_argument(
        "--no_tpu_cluster",
        action="store_false",
        dest="tpu_use_cluster",
        help="Should not be passed explicitly, this is for internal use only.",
    )
    tpu_args.add_argument(
        "--tpu_use_sudo",
        action="store_true",
        help="Whether to use `sudo` when running the TPU training script in each pod.",
    )
    tpu_args.add_argument(
        "--vm",
        type=str,
        action="append",
        help=(
            "List of single Compute VM instance names. "
            "If not provided we assume usage of instance groups. For TPU pods."
        ),
    )
    tpu_args.add_argument(
        "--env",
        type=str,
        action="append",
        help="List of environment variables to set on the Compute VM instances. For TPU pods.",
    )
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
        "--offload_optimizer_nvme_path",
        default=None,
        type=str,
        help="Decides Nvme Path to offload optimizer states (useful only when `use_deepspeed` flag is passed). "
        "If unspecified, will default to 'none'.",
    )
    deepspeed_args.add_argument(
        "--offload_param_nvme_path",
        default=None,
        type=str,
        help="Decides Nvme Path to offload parameters (useful only when `use_deepspeed` flag is passed). "
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
    deepspeed_args.add_argument(
        "--deepspeed_moe_layer_cls_names",
        default=None,
        type=str,
        help="comma-separated list of transformer MoE layer class names (case-sensitive) to wrap ,e.g, `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention,JetMoEBlock` ..."
        " (useful only when `use_deepspeed` flag is passed).",
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
        type=str,
        default="FULL_SHARD",
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
        "--fsdp_backward_prefetch",
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
    fsdp_args.add_argument(
        "--fsdp_forward_prefetch",
        default="false",
        type=str,
        help="If True, then FSDP explicitly prefetches the next upcoming "
        "all-gather while executing in the forward pass (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_use_orig_params",
        default="true",
        type=str,
        help="If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres."
        " (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_cpu_ram_efficient_loading",
        default="true",
        type=str,
        help="If True, only the first process loads the pretrained model checkoint while all other processes have empty weights. "
        "Only applicable for 🤗 Transformers. When using this, `--fsdp_sync_module_states` needs to True. "
        "(useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_sync_module_states",
        default="true",
        type=str,
        help="If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0."
        " (useful only when `use_fsdp` flag is passed).",
    )
    fsdp_args.add_argument(
        "--fsdp_activation_checkpointing",
        default="false",
        type=str,
        help="Decides Whether (true|false) intermediate activations are freed during the forward pass, and a checkpoint is left as a placeholder. (useful only when `use_fsdp` flag is passed).",
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

    # FP8 arguments
    fp8_args = parser.add_argument_group(
        "FP8 Arguments", "Arguments related to FP8 training (requires `--mixed_precision=fp8`)"
    )
    fp8_args.add_argument(
        "--fp8_backend",
        type=str,
        choices=["te", "msamp"],
        help="Choose a backend to train with FP8 (te: TransformerEngine, msamp: MS-AMP)",
    )
    fp8_args.add_argument(
        "--fp8_use_autocast_during_eval",
        default=False,
        action="store_true",
        help="Whether to use FP8 autocast during eval mode (useful only when `--fp8_backend=te` is passed). Generally better metrics are found when this is not passed.",
    )
    fp8_args.add_argument(
        "--fp8_margin",
        type=int,
        default=0,
        help="The margin to use for the gradient scaling (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_interval",
        type=int,
        default=1,
        help="The interval to use for how often the scaling factor is recomputed (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_format",
        type=str,
        default="E4M3",
        choices=["E4M3", "HYBRID"],
        help="The format to use for the FP8 recipe (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_amax_history_len",
        type=int,
        default=1024,
        help="The length of the history to use for the scaling factor computation (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_amax_compute_algo",
        type=str,
        default="most_recent",
        choices=["max", "most_recent"],
        help="The algorithm to use for the scaling factor computation. (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_override_linear_precision",
        type=lambda x: tuple(map(str_to_bool, x.split(","))),
        default=(False, False, False),
        help="Whether or not to execute `fprop`, `dgrad`, and `wgrad` GEMMS in higher precision. Should be passed in a comma-seperated string of booleans (useful only when `--fp8_backend=te` is passed).",
    )
    fp8_args.add_argument(
        "--fp8_opt_level",
        type=str,
        default="O2",
        choices=["O1", "O2"],
        help="What level of 8-bit collective communication should be used with MS-AMP (useful only when `--fp8_backend=msamp` is passed).",
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

    # MPI arguments
    mpirun_args = parser.add_argument_group("MPI Arguments", "Arguments related to mpirun for Multi-CPU")
    mpirun_args.add_argument(
        "--mpirun_hostfile",
        type=str,
        default=None,
        help="Location for a hostfile for using Accelerate to launch a multi-CPU training job with mpirun. This will "
        "get passed to the MPI --hostfile or -f parameter, depending on which MPI program is installed.",
    )
    mpirun_args.add_argument(
        "--mpirun_ccl",
        type=int,
        default=1,
        help="The number of oneCCL worker threads when using Accelerate to launch multi-CPU training with mpirun.",
    )

    # Other arguments of the training scripts
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER, help="Arguments of the training script.")

    if subparsers is not None:
        parser.set_defaults(func=launch_command)
    return parser


def simple_launcher(args):
    cmd, current_env = prepare_simple_launcher_cmd_env(args)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        if not args.quiet:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        else:
            sys.exit(1)


def multi_gpu_launcher(args):
    import torch.distributed.run as distrib_run

    current_env = prepare_multi_gpu_env(args)
    if not check_cuda_p2p_ib_support():
        message = "Using RTX 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled."
        warn = False
        if "NCCL_P2P_DISABLE" not in current_env:
            current_env["NCCL_P2P_DISABLE"] = "1"
            warn = True
        if "NCCL_IB_DISABLE" not in current_env:
            current_env["NCCL_IB_DISABLE"] = "1"
            warn = True
        if warn:
            logger.warning(message)

    debug = getattr(args, "debug", False)
    args = _filter_args(
        args,
        distrib_run.get_args_parser(),
        ["--training_script", args.training_script, "--training_script_args", args.training_script_args],
    )

    with patch_environment(**current_env):
        try:
            distrib_run.run(args)
        except Exception:
            if is_rich_available() and debug:
                console = get_console()
                console.print("\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]")
                console.print_exception(suppress=[__file__], show_locals=False)
            else:
                raise


def deepspeed_launcher(args):
    import torch.distributed.run as distrib_run

    if not is_deepspeed_available():
        raise ImportError("DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source.")
    else:
        from deepspeed.launcher.runner import DEEPSPEED_ENVIRONMENT_NAME

    cmd, current_env = prepare_deepspeed_cmd_env(args)
    if not check_cuda_p2p_ib_support():
        message = "Using RTX 4000 series which doesn't support faster communication speedups. Ensuring P2P and IB communications are disabled."
        warn = False
        if "NCCL_P2P_DISABLE" not in current_env:
            current_env["NCCL_P2P_DISABLE"] = "1"
            warn = True
        if "NCCL_IB_DISABLE" not in current_env:
            current_env["NCCL_IB_DISABLE"] = "1"
            warn = True
        if warn:
            logger.warning(message)

    if args.num_machines > 1 and args.deepspeed_multinode_launcher != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
        with open(DEEPSPEED_ENVIRONMENT_NAME, "a") as f:
            valid_env_items = convert_dict_to_env_variables(current_env)
            if len(valid_env_items) > 1:
                f.writelines(valid_env_items)

        process = subprocess.Popen(cmd, env=current_env)
        process.wait()
        if process.returncode != 0:
            if not args.quiet:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
            else:
                sys.exit(1)
    else:
        debug = getattr(args, "debug", False)
        args = _filter_args(
            args,
            distrib_run.get_args_parser(),
            ["--training_script", args.training_script, "--training_script_args", args.training_script_args],
        )
        with patch_environment(**current_env):
            try:
                distrib_run.run(args)
            except Exception:
                if is_rich_available() and debug:
                    console = get_console()
                    console.print("\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]")
                    console.print_exception(suppress=[__file__], show_locals=False)
                else:
                    raise


def tpu_launcher(args):
    import torch_xla.distributed.xla_multiprocessing as xmp

    if args.no_python:
        raise ValueError("--no_python cannot be used with TPU launcher")

    args, current_env = prepare_tpu(args, {})

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


def tpu_pod_launcher(args):
    from torch_xla.distributed import xla_dist

    current_env = {}
    args, current_env = prepare_tpu(args, current_env, True)
    debug = getattr(args, "debug", False)

    training_script = args.training_script
    training_script_args = args.training_script_args
    new_args = _filter_args(
        args, xla_dist.get_args_parser(), ["--tpu", args.tpu_name, "--positional", "", "--restart-tpuvm-pod-server"]
    )

    if args.tpu_use_sudo:
        new_cmd = ["sudo"]
    else:
        new_cmd = []

    new_cmd += [
        "accelerate-launch",
        "--tpu",
        "--no_tpu_cluster",
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--num_processes",
        str(args.num_processes),
        "--main_training_function",
        str(args.main_training_function),
        training_script,
    ] + training_script_args

    new_args.positional = new_cmd
    bad_flags = ""
    for arg in vars(new_args):
        if arg.startswith("docker_"):
            value = getattr(new_args, arg)
            if value != "" and value is not None:
                bad_flags += f'{arg}="{value}"\n'
    if bad_flags != "":
        raise ValueError(
            f"Docker containers are not supported for TPU pod launcher currently, please remove the following flags:\n{bad_flags}"
        )
    new_args.env = [f"{k}={v}" for k, v in current_env.items()]
    new_args.env.append("ACCELERATE_IN_TPU_POD=1")
    try:
        xla_dist.resolve_and_execute(new_args)
    except Exception:
        if is_rich_available() and debug:
            console = get_console()
            console.print("\n[bold red]Using --debug, `torch_xla.xla_dist` Stack Trace:[/bold red]")
            console.print_exception(suppress=[__file__], show_locals=False)
        else:
            raise


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

    args, sagemaker_inputs = prepare_sagemager_args_inputs(sagemaker_config, args)

    huggingface_estimator = HuggingFace(**args)

    huggingface_estimator.fit(inputs=sagemaker_inputs)
    print(f"You can find your model data at: {huggingface_estimator.model_data}")


def _validate_launch_command(args):
    # Sanity checks
    if sum([args.multi_gpu, args.cpu, args.tpu, args.use_deepspeed, args.use_fsdp]) > 1:
        raise ValueError(
            "You can only use one of `--cpu`, `--multi_gpu`, `--tpu`, `--use_deepspeed`, `--use_fsdp` at a time."
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
            and not args.tpu_use_cluster
            and not args.use_deepspeed
            and not args.use_fsdp
            and not args.use_megatron_lm
        ):
            args.use_deepspeed = defaults.distributed_type == DistributedType.DEEPSPEED
            args.multi_gpu = (
                True
                if defaults.distributed_type
                in (
                    DistributedType.MULTI_GPU,
                    DistributedType.MULTI_NPU,
                    DistributedType.MULTI_MLU,
                    DistributedType.MULTI_MUSA,
                    DistributedType.MULTI_XPU,
                )
                else False
            )
            args.tpu = defaults.distributed_type == DistributedType.XLA
            args.use_fsdp = defaults.distributed_type == DistributedType.FSDP
            args.use_megatron_lm = defaults.distributed_type == DistributedType.MEGATRON_LM
            args.tpu_use_cluster = defaults.tpu_use_cluster if args.tpu else False
        if args.gpu_ids is None:
            if defaults.gpu_ids is not None:
                args.gpu_ids = defaults.gpu_ids
            else:
                args.gpu_ids = "all"

        if args.multi_gpu and args.num_machines is None:
            args.num_machines = defaults.num_machines

        if len(args.gpu_ids.split(",")) < 2 and (args.gpu_ids != "all") and args.multi_gpu and args.num_machines <= 1:
            raise ValueError(
                "Less than two GPU ids were configured and tried to run on on multiple GPUs. "
                "Please ensure at least two are specified for `--gpu_ids`, or use `--gpu_ids='all'`."
            )
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
                    for k in defaults.dynamo_config:
                        setattr(args, k, defaults.dynamo_config[k])
                    for k in defaults.ipex_config:
                        setattr(args, k, defaults.ipex_config[k])
                    for k in defaults.mpirun_config:
                        setattr(args, k, defaults.mpirun_config[k])
                    continue

                # Those args are handled separately
                if (
                    name not in ["compute_environment", "mixed_precision", "distributed_type"]
                    and getattr(args, name, None) is None
                ):
                    setattr(args, name, attr)
        if not args.debug:
            args.debug = defaults.debug

        if not args.mixed_precision:
            if defaults.mixed_precision is None:
                args.mixed_precision = "no"
            else:
                args.mixed_precision = defaults.mixed_precision
                mp_from_config_flag = True
        else:
            if args.use_cpu or (args.use_xpu and torch.xpu.is_available()):
                native_amp = is_torch_version(">=", "1.10")
            else:
                native_amp = is_bf16_available(True)
            if (
                args.mixed_precision == "bf16"
                and not native_amp
                and not (args.tpu and is_torch_xla_available(check_is_tpu=True))
            ):
                raise ValueError("bf16 mixed precision requires PyTorch >= 1.10 and a supported device.")

        # Silently set the default here
        if args.dynamo_backend is None:
            args.dynamo_backend = "no"
        if args.num_processes == -1:
            raise ValueError("You need to manually pass in `--num_processes` using this config yaml.")
    else:
        if args.num_processes is None:
            if args.use_xpu and is_xpu_available():
                args.num_processes = torch.xpu.device_count()
            elif is_mlu_available():
                args.num_processes = torch.mlu.device_count()
            elif is_musa_available():
                args.num_processes = torch.musa.device_count()
            elif is_npu_available():
                args.num_processes = torch.npu.device_count()
            else:
                args.num_processes = torch.cuda.device_count()
            warned.append(f"\t`--num_processes` was set to a value of `{args.num_processes}`")
        if args.debug is None:
            args.debug = False
        if (
            not args.multi_gpu
            and args.num_processes > 1
            and (
                (args.use_xpu and is_xpu_available() and torch.xpu.device_count() > 1)
                or (is_mlu_available() and torch.mlu.device_count() > 1)
                or (is_musa_available() and torch.musa.device_count() > 1)
                or (is_npu_available() and torch.npu.device_count() > 1)
                or (torch.cuda.device_count() > 1)
            )
        ):
            warned.append(
                "\t\tMore than one GPU was found, enabling multi-GPU training.\n"
                "\t\tIf this was unintended please pass in `--num_processes=1`."
            )
            args.multi_gpu = True
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
    if args.debug:
        logger.debug("Running script in debug mode, expect distributed operations to be slightly slower.")

    is_aws_env_disabled = defaults is None or (
        defaults is not None and defaults.compute_environment != ComputeEnvironment.AMAZON_SAGEMAKER
    )
    if is_aws_env_disabled and args.num_cpu_threads_per_process is None:
        args.num_cpu_threads_per_process = get_int_from_env(["OMP_NUM_THREADS"], 1)
        if args.use_cpu and args.num_processes >= 1 and get_int_from_env(["OMP_NUM_THREADS"], 0) == 0:
            local_size = get_int_from_env(
                ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"],
                max(int(args.num_processes / args.num_machines), 1),
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
    return args, defaults, mp_from_config_flag


def launch_command(args):
    args, defaults, mp_from_config_flag = _validate_launch_command(args)
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
        if args.tpu_use_cluster:
            tpu_pod_launcher(args)
        else:
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
