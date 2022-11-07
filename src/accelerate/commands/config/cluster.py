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

import os

from ...utils import ComputeEnvironment, DistributedType, is_deepspeed_available, is_transformers_available
from ...utils.constants import (
    DEEPSPEED_MULTINODE_LAUNCHERS,
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    FSDP_STATE_DICT_TYPE,
)
from .config_args import ClusterConfig
from .config_utils import _ask_field, _convert_distributed_mode, _convert_yes_no_to_bool


def get_cluster_input():
    distributed_type = _ask_field(
        "Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): ",
        _convert_distributed_mode,
        error_message="Please enter 0, 1, 2, 3 or 4.",
    )

    machine_rank = 0
    num_machines = 1
    num_processes = 1
    gpu_ids = None
    main_process_ip = None
    main_process_port = None
    rdzv_backend = "static"
    same_network = True
    tpu_name = None
    tpu_zone = None
    commands = None
    command_file = None
    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.MULTI_CPU]:
        num_machines = _ask_field(
            "How many different machines will you use (use more than 1 for multi-node training)? [1]: ",
            lambda x: int(x),
            default=1,
        )
        if num_machines > 1:
            machine_rank = _ask_field(
                "What is the rank of this machine (from 0 to the number of machines - 1 )? [0]: ",
                lambda x: int(x),
                default=0,
            )
            main_process_ip = _ask_field(
                "What is the IP address of the machine that will host the main process? ",
            )
            main_process_port = _ask_field(
                "What is the port you will use to communicate with the main process? ",
                lambda x: int(x),
            )
            same_network = _ask_field(
                "Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: ",
                _convert_yes_no_to_bool,
                default=True,
                error_message="Please enter yes or no.",
            )
            if not same_network:
                rdzv_backend = _ask_field(
                    "What rendezvous backend will you use? ('static', 'c10d', ...): ", default="static"
                )

    if distributed_type == DistributedType.NO:
        use_cpu = _ask_field(
            "Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
    elif distributed_type == DistributedType.MULTI_CPU:
        use_cpu = True
    else:
        use_cpu = False

    deepspeed_config = {}
    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.NO]:
        use_deepspeed = _ask_field(
            "Do you want to use DeepSpeed? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
        if use_deepspeed:
            distributed_type = DistributedType.DEEPSPEED
            assert (
                is_deepspeed_available()
            ), "DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source"

        if distributed_type == DistributedType.DEEPSPEED:
            use_deepspeed_config = _ask_field(
                "Do you want to specify a json file to a DeepSpeed config? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            if use_deepspeed_config:
                deepspeed_config["deepspeed_config_file"] = _ask_field(
                    "Please enter the path to the json DeepSpeed config file: ",
                    lambda x: str(x),
                    default="none",
                )
            else:
                deepspeed_config["zero_stage"] = _ask_field(
                    "What should be your DeepSpeed's ZeRO optimization stage (0, 1, 2, 3)? [2]: ",
                    lambda x: int(x),
                    default=2,
                )

                if deepspeed_config["zero_stage"] >= 2:
                    deepspeed_config["offload_optimizer_device"] = _ask_field(
                        "Where to offload optimizer states? [none/cpu/nvme]: ",
                        lambda x: str(x),
                        default="none",
                    )
                    deepspeed_config["offload_param_device"] = _ask_field(
                        "Where to offload parameters? [none/cpu/nvme]: ",
                        lambda x: str(x),
                        default="none",
                    )
                deepspeed_config["gradient_accumulation_steps"] = _ask_field(
                    "How many gradient accumulation steps you're passing in your script? [1]: ",
                    lambda x: int(x),
                    default=1,
                )
                use_gradient_clipping = _ask_field(
                    "Do you want to use gradient clipping? [yes/NO]: ",
                    _convert_yes_no_to_bool,
                    default=False,
                    error_message="Please enter yes or no.",
                )
                if use_gradient_clipping:
                    deepspeed_config["gradient_clipping"] = _ask_field(
                        "What is the gradient clipping value? [1.0]: ",
                        lambda x: float(x),
                        default=1.0,
                    )
                if deepspeed_config["zero_stage"] == 3:
                    deepspeed_config["zero3_save_16bit_model"] = _ask_field(
                        "Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: ",
                        _convert_yes_no_to_bool,
                        default=False,
                        error_message="Please enter yes or no.",
                    )
            deepspeed_config["zero3_init_flag"] = _ask_field(
                "Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            if deepspeed_config["zero3_init_flag"]:
                if not is_transformers_available():
                    raise Exception(
                        "When `zero3_init_flag` is set, it requires Transformers to be installed. "
                        "Please run `pip3 install transformers`."
                    )

            if num_machines > 1:
                launcher_query = "Which Type of launcher do you want to use "
                for i, launcher in enumerate(DEEPSPEED_MULTINODE_LAUNCHERS):
                    launcher_query += f"[{i}] {launcher}, "
                launcher_query = launcher_query[:-2] + ")? [0]: "
                deepspeed_config["deepspeed_multinode_launcher"] = _ask_field(
                    launcher_query,
                    lambda x: DEEPSPEED_MULTINODE_LAUNCHERS[int(x)],
                    default=DEEPSPEED_MULTINODE_LAUNCHERS[0],
                )

                if deepspeed_config["deepspeed_multinode_launcher"] != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
                    deepspeed_config["deepspeed_hostfile"] = _ask_field(
                        "DeepSpeed configures multi-node compute resources with hostfile. "
                        "Each row is of the format `hostname slots=[num_gpus]`, e.g., `localhost slots=2`; "
                        "for more information please refer official [documentation]"
                        "(https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). "
                        "Please specify the location of hostfile: ",
                        lambda x: str(x),
                    )

                    is_exclusion_filter = _ask_field(
                        "Do you want to specify exclusion filter string? [yes/NO]: ",
                        _convert_yes_no_to_bool,
                        default=False,
                        error_message="Please enter yes or no.",
                    )
                    if is_exclusion_filter:
                        deepspeed_config["deepspeed_exclusion_filter"] = _ask_field(
                            "DeepSpeed exclusion filter string: ",
                            lambda x: str(x),
                        )

                    is_inclusion_filter = _ask_field(
                        "Do you want to specify inclusion filter string? [yes/NO]: ",
                        _convert_yes_no_to_bool,
                        default=False,
                        error_message="Please enter yes or no.",
                    )
                    if is_inclusion_filter:
                        deepspeed_config["deepspeed_inclusion_filter"] = _ask_field(
                            "DeepSpeed inclusion filter string: ",
                            lambda x: str(x),
                        )

    fsdp_config = {}
    if distributed_type in [DistributedType.MULTI_GPU]:
        use_fsdp = _ask_field(
            "Do you want to use FullyShardedDataParallel? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
        if use_fsdp:
            distributed_type = DistributedType.FSDP
        if distributed_type == DistributedType.FSDP:
            sharding_strategy_query = "What should be your sharding strategy ("
            for i, strategy in enumerate(FSDP_SHARDING_STRATEGY):
                sharding_strategy_query += f"[{i+1}] {strategy}, "
            sharding_strategy_query = sharding_strategy_query[:-2] + ")? [1]: "
            fsdp_config["fsdp_sharding_strategy"] = _ask_field(
                sharding_strategy_query,
                lambda x: int(x),
                default=1,
            )
            fsdp_config["fsdp_offload_params"] = _ask_field(
                "Do you want to offload parameters and gradients to CPU? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            fsdp_wrap_query = "What should be your auto wrap policy ("
            for i, wrap_policy in enumerate(FSDP_AUTO_WRAP_POLICY):
                fsdp_wrap_query += f"[{i}] {wrap_policy}, "
            fsdp_wrap_query = fsdp_wrap_query[:-2] + ")? [0]: "
            fsdp_config["fsdp_auto_wrap_policy"] = _ask_field(
                fsdp_wrap_query,
                lambda x: FSDP_AUTO_WRAP_POLICY[int(x)],
                default="TRANSFORMER_BASED_WRAP",
            )
            if fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[0]:
                fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = _ask_field(
                    "What is the transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` ...? : ",
                    lambda x: str(x),
                )
            elif fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[1]:
                fsdp_config["fsdp_min_num_params"] = _ask_field(
                    "What should be your FSDP's minimum number of parameters for Default Auto Wrapping Policy? [1e8]: ",
                    lambda x: int(x),
                    default=1e8,
                )
            fsdp_backward_prefetch_query = "What should be your FSDP's backward prefetch policy ("
            for i, backward_prefetch_policy in enumerate(FSDP_BACKWARD_PREFETCH):
                fsdp_backward_prefetch_query += f"[{i}] {backward_prefetch_policy}, "
            fsdp_backward_prefetch_query = fsdp_backward_prefetch_query[:-2] + ")? [0]: "
            fsdp_config["fsdp_backward_prefetch_policy"] = _ask_field(
                fsdp_backward_prefetch_query,
                lambda x: FSDP_BACKWARD_PREFETCH[int(x)],
                default="BACKWARD_PRE",
            )
            fsdp_state_dict_type_query = "What should be your FSDP's state dict type ("
            for i, state_dict_type in enumerate(FSDP_STATE_DICT_TYPE):
                fsdp_state_dict_type_query += f"[{i}] {state_dict_type}, "
            fsdp_state_dict_type_query = fsdp_state_dict_type_query[:-2] + ")? [0]: "
            fsdp_config["fsdp_state_dict_type"] = _ask_field(
                fsdp_state_dict_type_query,
                lambda x: FSDP_STATE_DICT_TYPE[int(x)],
                default="FULL_STATE_DICT",
            )

    megatron_lm_config = {}
    if distributed_type in [DistributedType.MULTI_GPU]:
        use_megatron_lm = _ask_field(
            "Do you want to use Megatron-LM ? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
        if use_megatron_lm:
            distributed_type = DistributedType.MEGATRON_LM
        if distributed_type == DistributedType.MEGATRON_LM:
            prefix = "megatron_lm_"
            megatron_lm_config[prefix + "tp_degree"] = _ask_field(
                "What is the Tensor Parallelism degree/size? [1]:",
                lambda x: int(x),
                default=1,
                error_message="Please enter an integer.",
            )
            if megatron_lm_config[prefix + "tp_degree"] > 1:
                megatron_lm_config[prefix + "sequence_parallelism"] = _ask_field(
                    "Do you want to enable Sequence Parallelism? [YES/no]: ",
                    _convert_yes_no_to_bool,
                    default=True,
                    error_message="Please enter yes or no.",
                )

            megatron_lm_config[prefix + "pp_degree"] = _ask_field(
                "What is the Pipeline Parallelism degree/size? [1]:",
                lambda x: int(x),
                default=1,
                error_message="Please enter an integer.",
            )
            if megatron_lm_config[prefix + "pp_degree"] > 1:
                megatron_lm_config[prefix + "num_micro_batches"] = _ask_field(
                    "What is the number of micro-batches? [1]:",
                    lambda x: int(x),
                    default=1,
                    error_message="Please enter an integer.",
                )

            megatron_lm_config[prefix + "recompute_activations"] = _ask_field(
                "Do you want to enable selective activation recomputation? [YES/no]: ",
                _convert_yes_no_to_bool,
                default=True,
                error_message="Please enter yes or no.",
            )

            megatron_lm_config[prefix + "use_distributed_optimizer"] = _ask_field(
                "Do you want to use distributed optimizer "
                "which shards optimizer state and gradients across data pralellel ranks? [YES/no]: ",
                _convert_yes_no_to_bool,
                default=True,
                error_message="Please enter yes or no.",
            )

            megatron_lm_config[prefix + "gradient_clipping"] = _ask_field(
                "What is the gradient clipping value based on global L2 Norm (0 to disable)? [1.0]: ",
                lambda x: float(x),
                default=1.0,
            )

    if distributed_type == DistributedType.TPU:
        main_training_function = _ask_field(
            "What is the name of the function in your script that should be launched in all parallel scripts? [main]: ",
            default="main",
        )
        use_cluster = _ask_field(
            "Are you using a TPU cluster? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
        if use_cluster:
            tpu_name = _ask_field(
                "What is the name of your TPU cluster? ",
                default=None,
                error_message="Please enter the name of your TPU cluster.",
            )
            tpu_zone = _ask_field(
                "What is the zone of your TPU cluster? ",
                default=None,
                error_message="Please enter the zone of your TPU cluster.",
            )
            run_commands = _ask_field(
                "Do you have code you wish to run on startup in each pod? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            if run_commands:
                use_command_file = _ask_field(
                    "Is this code located in a bash script? [yes/NO]: ",
                    _convert_yes_no_to_bool,
                    default=False,
                    error_message="Please enter yes or no.",
                )
                if use_command_file:
                    command_file = _ask_field(
                        "What is the path to your bash script? ",
                        default=None,
                        error_message="Please enter the path to your bash script.",
                    )
                    command_file = os.path.abspath(command_file)
                else:
                    print("Please enter each command seperately you wish to run on startup in each pod.")
                    commands = []
                    another_command = True
                    while another_command:
                        commands.append(
                            _ask_field(
                                "Please enter a single command to be ran ",
                                default=None,
                                error_message="Please enter the commands you wish to run on startup in each pod as a single string.",
                            )
                        )
                        another_command = _ask_field(
                            "Do you wish to add another command? [yes/NO]: ",
                            _convert_yes_no_to_bool,
                            default=False,
                            error_message="Please enter yes or no.",
                        )

    else:
        main_training_function = "main"

    if distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_GPU, DistributedType.TPU]:
        machine_type = str(distributed_type).split(".")[1].replace("MULTI_", "")
        if machine_type == "TPU":
            machine_type += " cores"
        else:
            machine_type += "(s)"
        num_processes = _ask_field(
            f"How many {machine_type} should be used for distributed training? [1]:",
            lambda x: int(x),
            default=1,
            error_message="Please enter an integer.",
        )
    elif distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
        num_processes = _ask_field(
            "How many GPU(s) should be used for distributed training? [1]:",
            lambda x: int(x),
            default=1,
            error_message="Please enter an integer.",
        )
    else:
        num_processes = 1

    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.NO] and not use_cpu:
        gpu_ids = _ask_field(
            "What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:",
            default="all",
        )

    if distributed_type != DistributedType.TPU:
        if distributed_type == DistributedType.DEEPSPEED and use_deepspeed_config:
            mixed_precision = "no"
        else:
            mixed_precision = _ask_field(
                "Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: ",
                lambda x: str(x).lower(),
                default="no",
            )
    else:
        mixed_precision = "no"

    downcast_bf16 = "no"
    if distributed_type == DistributedType.TPU and mixed_precision == "bf16":
        downcast_bf16 = _ask_field(
            "Should `torch.float` be cast as `bfloat16` and `torch.double` remain `float32` on TPUs?", default="no"
        )

    return ClusterConfig(
        compute_environment=ComputeEnvironment.LOCAL_MACHINE,
        distributed_type=distributed_type,
        num_processes=num_processes,
        gpu_ids=gpu_ids,
        mixed_precision=mixed_precision,
        downcast_bf16=downcast_bf16,
        machine_rank=machine_rank,
        num_machines=num_machines,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
        main_training_function=main_training_function,
        deepspeed_config=deepspeed_config,
        fsdp_config=fsdp_config,
        megatron_lm_config=megatron_lm_config,
        use_cpu=use_cpu,
        rdzv_backend=rdzv_backend,
        same_network=same_network,
        tpu_name=tpu_name,
        tpu_zone=tpu_zone,
        commands=commands,
        command_file=command_file,
    )
