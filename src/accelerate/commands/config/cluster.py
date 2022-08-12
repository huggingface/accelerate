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

from ...utils import ComputeEnvironment, DistributedType, is_deepspeed_available, is_transformers_available
from ...utils.constants import (
    DEEPSPEED_MULTINODE_LAUNCHERS,
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    FSDP_STATE_DICT_TYPE,
    RICH_COLORS,
)
from ...utils.rich import _ask_field
from .config_args import ClusterConfig
from .config_utils import _convert_distributed_mode


def get_cluster_input():
    config = {
        "machine_rank": 0,
        "num_machines": 1,
        "main_process_ip": None,
        "main_process_port": None,
        "main_training_function": "main",
        "num_processes": 1,
        "mixed_precision": "no",
        "downcast_bf16": "no",
        "use_cpu": False,
        "deepspeed_config": {},
        "fsdp_config": {},
    }

    distributed_type = _ask_field(
        f"Which type of machine are you using? ([{RICH_COLORS[0]}][0] No distributed training[/{RICH_COLORS[0]}], [{RICH_COLORS[1]}][1] multi-CPU[/{RICH_COLORS[1]}], [{RICH_COLORS[2]}][2] multi-GPU[/{RICH_COLORS[2]}], [{RICH_COLORS[3]}][3] TPU[/#FE6100], [#FFB000][4] MPS[/#FFB000])",
        "int",
        choices=["0", "1", "2", "3", "4"],
    )
    config["distributed_type"] = _convert_distributed_mode(distributed_type)

    if config["distributed_type"] in [DistributedType.MULTI_GPU, DistributedType.MULTI_CPU]:
        config["num_machines"] = _ask_field(
            "How many different machines will you use? (use more than 1 for multi-node training)",
            "int",
            default=config["num_machines"],
        )
        if config["num_machines"] > 1:
            config["machine_rank"] = _ask_field(
                "What is the rank of this machine?",
                "int",
                default=config["machine_rank"],
                choices=[*map(str, range(0, config["num_machines"]))],
            )
            config["main_process_ip"] = _ask_field(
                "What is the IP address of the machine that will host the main process",
            )
            config["main_process_port"] = _ask_field(
                "What is the port you will use to communicate with the main process",
            )

    if config["distributed_type"] == DistributedType.NO:
        config["use_cpu"] = _ask_field(
            "Do you want to run your training on CPU only (even if a GPU is available)",
            "bool",
        )
    elif config["distributed_type"] == DistributedType.MULTI_CPU:
        config["use_cpu"] = True

    if config["distributed_type"] in [DistributedType.MULTI_GPU, DistributedType.NO]:
        use_deepspeed = _ask_field("Do you want to use DeepSpeed?", "bool")
        if use_deepspeed:
            deepspeed_config = {}
            config["distributed_type"] = DistributedType.DEEPSPEED
            assert (
                is_deepspeed_available()
            ), "DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source"

        if config["distributed_type"] == DistributedType.DEEPSPEED:
            use_deepspeed_config = _ask_field(
                "Do you want to specify a json file to a DeepSpeed config?",
                "bool",
            )
            if use_deepspeed_config:
                deepspeed_config["deepspeed_config_file"] = _ask_field(
                    "Please enter the path to the json DeepSpeed config file",
                    default="none",
                )
            else:
                deepspeed_config["zero_stage"] = _ask_field(
                    "What should be your DeepSpeed's ZeRO optimization stage?",
                    "int",
                    choices=[*map(str, range(0, 4))],
                    default=2,
                )

                if deepspeed_config["zero_stage"] >= 2:
                    deepspeed_config["offload_optimizer_device"] = _ask_field(
                        "Where should optimizer states be offloaded to?",
                        choices=["none", "cpu", "nvme"],
                        default="none",
                    )
                    deepspeed_config["offload_param_device"] = _ask_field(
                        "Where should parameters be offloaded to?",
                        choices=["none", "cpu", "nvme"],
                        default="none",
                    )
                deepspeed_config["gradient_accumulation_steps"] = _ask_field(
                    "How many gradient accumulation steps are you passing in your script?",
                    "int",
                    default=1,
                )
                use_gradient_clipping = _ask_field(
                    "Do you want to use gradient clipping?",
                    "bool",
                )
                if use_gradient_clipping:
                    deepspeed_config["gradient_clipping"] = _ask_field(
                        "What is the gradient clipping value?",
                        "float",
                        default=1.0,
                    )
                if deepspeed_config["zero_stage"] == 3:
                    deepspeed_config["zero3_save_16bit_model"] = _ask_field(
                        "Do you want to save 16-bit model weights when using ZeRO Stage-3?",
                        "bool",
                    )
            deepspeed_config["zero3_init_flag"] = _ask_field(
                "Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models?",
                "bool",
            )
            if deepspeed_config["zero3_init_flag"]:
                if not is_transformers_available():
                    raise Exception(
                        "When `zero3_init_flag` is set, it requires Transformers to be installed. "
                        "Please run `pip3 install transformers`."
                    )

            if config["num_machines"] > 1:
                launcher_query = "Which Type of launcher do you want to use? ("
                for i, launcher in enumerate(DEEPSPEED_MULTINODE_LAUNCHERS):
                    launcher_query += f"[{RICH_COLORS[i]}][{i}] {launcher}[/{RICH_COLORS[i]}], "
                launcher_query = launcher_query[:-2] + ")"
                deepspeed_config["deepspeed_multinode_launcher"] = DEEPSPEED_MULTINODE_LAUNCHERS[
                    _ask_field(launcher_query, "int", default=0, choices=[*map(str, range(3))])
                ]

                if deepspeed_config["deepspeed_multinode_launcher"] != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
                    deepspeed_config["deepspeed_hostfile"] = _ask_field(
                        "DeepSpeed configures multi-node compute resources with hostfile. "
                        "Each row is of the format `hostname slots=[num_gpus]`, e.g., `localhost slots=2`; "
                        "for more information please refer official [documentation]"
                        "(https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). "
                        "Please specify the location of hostfile"
                    )

                    is_exclusion_filter = _ask_field("Do you want to specify exclusion filter string?", "bool")
                    if is_exclusion_filter:
                        deepspeed_config["deepspeed_exclusion_filter"] = _ask_field(
                            "DeepSpeed exclusion filter string"
                        )

                    is_inclusion_filter = _ask_field(
                        "Do you want to specify inclusion filter string?",
                        "bool",
                    )
                    if is_inclusion_filter:
                        deepspeed_config["deepspeed_inclusion_filter"] = _ask_field(
                            "DeepSpeed inclusion filter string: "
                        )
            config["deepspeed_config"] = deepspeed_config

    if config["distributed_type"] in [DistributedType.MULTI_GPU]:
        use_fsdp = _ask_field(
            "Do you want to use FullyShardedDataParallel?",
            "bool",
        )
        if use_fsdp:
            fsdp_config = {}
            config["distributed_type"] = DistributedType.FSDP
        if config["distributed_type"] == DistributedType.FSDP:
            sharding_strategy_query = "What should be your sharding strategy? ("
            for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
                sharding_strategy_query += f"[{RICH_COLORS[i]}][{i+1}] {strategy}[/{RICH_COLORS[i]}], "
            sharding_strategy_query = sharding_strategy_query[:-2] + ")"
            fsdp_config["fsdp_sharding_strategy"] = _ask_field(
                sharding_strategy_query,
                "int",
                default=1,
            )
            fsdp_config["fsdp_offload_params"] = _ask_field(
                "Do you want to offload parameters and gradients to CPU?",
                "bool",
            )
            fsdp_wrap_query = "What should be your auto wrap policy? ("
            for i, wrap_policy in enumerate(FSDP_AUTO_WRAP_POLICY):
                fsdp_wrap_query += f"[{RICH_COLORS[i]}][{i}]{wrap_policy}[/{RICH_COLORS[i]}], "
            fsdp_wrap_query = fsdp_wrap_query[:-2] + ")"
            fsdp_config["fsdp_auto_wrap_policy"] = FSDP_AUTO_WRAP_POLICY[
                _ask_field(
                    fsdp_wrap_query,
                    "int",
                    choices=[*map(str, range(3))],
                )
            ]
            if fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[0]:
                fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = _ask_field(
                    "What is the transformer layer class name (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`, `T5Block` ...?",
                )
            elif fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[1]:
                fsdp_config["fsdp_min_num_params"] = _ask_field(
                    "What should be your FSDP's minimum number of parameters for Default Auto Wrapping Policy?",
                    "int",
                    default=1e8,
                )
            fsdp_backward_prefetch_query = "What should be your FSDP's backward prefetch policy? ("
            for i, backward_prefetch_policy in enumerate(FSDP_BACKWARD_PREFETCH):
                fsdp_backward_prefetch_query += (
                    f"[{RICH_COLORS[i]}][{i}] {backward_prefetch_policy}[/{RICH_COLORS[i]}][{i}], "
                )
            fsdp_backward_prefetch_query = fsdp_backward_prefetch_query[:-2] + ")"
            fsdp_config["fsdp_backward_prefetch_policy"] = FSDP_BACKWARD_PREFETCH[
                _ask_field(
                    fsdp_backward_prefetch_query,
                    "int",
                    choices=[*map(str, range(3))],
                )
            ]
            fsdp_state_dict_type_query = "What should be your FSDP's state dict type? ("
            for i, state_dict_type in enumerate(FSDP_STATE_DICT_TYPE):
                fsdp_state_dict_type_query += f"[{RICH_COLORS[i]}][{i}] {state_dict_type}[/{RICH_COLORS[i]}], "
            fsdp_state_dict_type_query = fsdp_state_dict_type_query[:-2] + ")"
            fsdp_config["fsdp_state_dict_type"] = FSDP_STATE_DICT_TYPE[
                _ask_field(fsdp_state_dict_type_query, "int", choices=[*map(str, range(3))])
            ]
            config["fsdp_config"] = fsdp_config

    if config["distributed_type"] == DistributedType.TPU:
        config["main_training_function"] = _ask_field(
            "What is the name of the function in your script that should be launched in all parallel scripts?",
            default="main",
        )

    if config["distributed_type"] in [DistributedType.MULTI_CPU, DistributedType.MULTI_GPU, DistributedType.TPU]:
        machine_type = str(config["distributed_type"]).split(".")[1].replace("MULTI_", "")
        if machine_type == "TPU":
            machine_type += " cores"
            default = 8
        else:
            machine_type += "(s)"
            default = 1
        config["num_processes"] = _ask_field(
            f"How many {machine_type} should be used for distributed training?",
            "int",
            default=default,
        )
    elif config["distributed_type"] in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
        config["num_processes"] = _ask_field(
            "How many GPU(s) should be used for distributed training?",
            "int",
            default=1,
        )
    if not (config["distributed_type"] == DistributedType.DEEPSPEED and use_deepspeed_config):
        config["mixed_precision"] = _ask_field(
            "Do you wish to use FP16 or BF16 (mixed precision)?",
            choices=["no", "fp16", "bf16"],
            default="no",
        )

    if config["distributed_type"] == DistributedType.TPU and config["mixed_precision"] == "bf16":
        config["downcast_bf16"] = _ask_field(
            "Should `torch.float` be cast as `bfloat16` and `torch.double` remain `float32` on TPUs?",
            "bool",
        )

    return ClusterConfig(compute_environment=ComputeEnvironment.LOCAL_MACHINE, **config)
