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

from ...utils import (
    ComputeEnvironment,
    DistributedType,
    is_deepspeed_available,
    is_mps_available,
    is_transformers_available,
)
from ...utils.constants import (
    DEEPSPEED_MULTINODE_LAUNCHERS,
    FSDP_AUTO_WRAP_POLICY,
    FSDP_BACKWARD_PREFETCH,
    FSDP_SHARDING_STRATEGY,
    FSDP_STATE_DICT_TYPE,
    TORCH_DYNAMO_MODES,
)
from .config_args import ClusterConfig
from .config_utils import (
    DYNAMO_BACKENDS,
    _ask_field,
    _ask_options,
    _convert_distributed_mode,
    _convert_dynamo_backend,
    _convert_mixed_precision,
    _convert_yes_no_to_bool,
)


def get_cluster_input():
    distributed_type = _ask_options(
        "Which type of machine are you using?",
        ["No distributed training", "multi-CPU", "multi-GPU", "TPU"],
        _convert_distributed_mode,
    )

    machine_rank = 0
    num_machines = 1
    num_processes = 1
    gpu_ids = None
    main_process_ip = None
    main_process_port = None
    rdzv_backend = "static"
    same_network = True

    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.MULTI_CPU]:
        num_machines = _ask_field(
            "How many different machines will you use (use more than 1 for multi-node training)? [1]: ",
            int,
            default=1,
        )
        if num_machines > 1:
            machine_rank = _ask_options(
                "What is the rank of this machine?",
                list(range(num_machines)),
                int,
            )
            main_process_ip = _ask_field(
                "What is the IP address of the machine that will host the main process? ",
            )
            main_process_port = _ask_field(
                "What is the port you will use to communicate with the main process? ",
                int,
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
            "Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
    elif distributed_type == DistributedType.MULTI_CPU:
        use_cpu = True
    else:
        use_cpu = False

    dynamo_config = {}
    use_dynamo = _ask_field(
        "Do you wish to optimize your script with torch dynamo?[yes/NO]:",
        _convert_yes_no_to_bool,
        default=False,
        error_message="Please enter yes or no.",
    )
    if use_dynamo:
        prefix = "dynamo_"
        dynamo_config[prefix + "backend"] = _ask_options(
            "Which dynamo backend would you like to use?",
            [x.lower() for x in DYNAMO_BACKENDS],
            _convert_dynamo_backend,
            default=2,
        )
        use_custom_options = _ask_field(
            "Do you want to customize the defaults sent to torch.compile? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )

        if use_custom_options:
            dynamo_config[prefix + "mode"] = _ask_options(
                "Which mode do you want to use?",
                TORCH_DYNAMO_MODES,
                lambda x: TORCH_DYNAMO_MODES[int(x)],
                default=0,
            )
            dynamo_config[prefix + "use_fullgraph"] = _ask_field(
                "Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            dynamo_config[prefix + "use_dynamic"] = _ask_field(
                "Do you want to enable dynamic shape tracing? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )

    use_mps = not use_cpu and is_mps_available()
    deepspeed_config = {}
    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.NO] and not use_mps:
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
                    str,
                    default="none",
                )
            else:
                deepspeed_config["zero_stage"] = _ask_options(
                    "What should be your DeepSpeed's ZeRO optimization stage?",
                    [0, 1, 2, 3],
                    int,
                    default=2,
                )

                deepspeed_devices = ["none", "cpu", "nvme"]
                if deepspeed_config["zero_stage"] >= 2:
                    deepspeed_config["offload_optimizer_device"] = _ask_options(
                        "Where to offload optimizer states?", deepspeed_devices, lambda x: deepspeed_devices[int(x)]
                    )
                    deepspeed_config["offload_param_device"] = _ask_options(
                        "Where to offload parameters?", deepspeed_devices, lambda x: deepspeed_devices[int(x)]
                    )
                deepspeed_config["gradient_accumulation_steps"] = _ask_field(
                    "How many gradient accumulation steps you're passing in your script? [1]: ",
                    int,
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
                        float,
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
                launcher_query = "Which Type of launcher do you want to use?"
                deepspeed_config["deepspeed_multinode_launcher"] = _ask_options(
                    launcher_query,
                    DEEPSPEED_MULTINODE_LAUNCHERS,
                    lambda x: DEEPSPEED_MULTINODE_LAUNCHERS[int(x)],
                )

                if deepspeed_config["deepspeed_multinode_launcher"] != DEEPSPEED_MULTINODE_LAUNCHERS[1]:
                    deepspeed_config["deepspeed_hostfile"] = _ask_field(
                        "DeepSpeed configures multi-node compute resources with hostfile. "
                        "Each row is of the format `hostname slots=[num_gpus]`, e.g., `localhost slots=2`; "
                        "for more information please refer official [documentation]"
                        "(https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). "
                        "Please specify the location of hostfile: ",
                        str,
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
                            str,
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
                            str,
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
            sharding_strategy_query = "What should be your sharding strategy?"
            fsdp_config["fsdp_sharding_strategy"] = _ask_options(
                sharding_strategy_query,
                FSDP_SHARDING_STRATEGY,
                lambda x: int(x) + 1,
                default=1,
            )
            fsdp_config["fsdp_offload_params"] = _ask_field(
                "Do you want to offload parameters and gradients to CPU? [yes/NO]: ",
                _convert_yes_no_to_bool,
                default=False,
                error_message="Please enter yes or no.",
            )
            fsdp_wrap_query = "What should be your auto wrap policy?"
            fsdp_config["fsdp_auto_wrap_policy"] = _ask_options(
                fsdp_wrap_query,
                FSDP_AUTO_WRAP_POLICY,
                lambda x: FSDP_AUTO_WRAP_POLICY[int(x)],
            )
            if fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[0]:
                fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = _ask_field(
                    "Specify the comma-separated list of transformer layer class names (case-sensitive) to wrap ,e.g, :"
                    "`BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput` ...? : ",
                    str,
                )
            elif fsdp_config["fsdp_auto_wrap_policy"] == FSDP_AUTO_WRAP_POLICY[1]:
                fsdp_config["fsdp_min_num_params"] = _ask_field(
                    "What should be your FSDP's minimum number of parameters for Default Auto Wrapping Policy? [1e8]: ",
                    int,
                    default=1e8,
                )
            fsdp_backward_prefetch_query = "What should be your FSDP's backward prefetch policy?"
            fsdp_config["fsdp_backward_prefetch_policy"] = _ask_options(
                fsdp_backward_prefetch_query,
                FSDP_BACKWARD_PREFETCH,
                lambda x: FSDP_BACKWARD_PREFETCH[int(x)],
            )
            fsdp_state_dict_type_query = "What should be your FSDP's state dict type?"
            fsdp_config["fsdp_state_dict_type"] = _ask_options(
                fsdp_state_dict_type_query,
                FSDP_STATE_DICT_TYPE,
                lambda x: FSDP_STATE_DICT_TYPE[int(x)],
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
                int,
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
                int,
                default=1,
                error_message="Please enter an integer.",
            )
            if megatron_lm_config[prefix + "pp_degree"] > 1:
                megatron_lm_config[prefix + "num_micro_batches"] = _ask_field(
                    "What is the number of micro-batches? [1]:",
                    int,
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
                float,
                default=1.0,
            )
    # TPU specific defaults
    tpu_commands = None
    tpu_command_file = None
    tpu_downcast_bf16 = "no"
    tpu_env = []
    tpu_name = None
    tpu_vm = None
    tpu_zone = None
    tpu_use_sudo = False
    tpu_use_cluster = False

    if distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_GPU, DistributedType.TPU]:
        machine_type = str(distributed_type).split(".")[1].replace("MULTI_", "")
        if machine_type == "TPU":
            machine_type += " cores"
        else:
            machine_type += "(s)"
        num_processes = _ask_field(
            f"How many {machine_type} should be used for distributed training? [1]:",
            int,
            default=1,
            error_message="Please enter an integer.",
        )
    elif distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
        num_processes = _ask_field(
            "How many GPU(s) should be used for distributed training? [1]:",
            int,
            default=1,
            error_message="Please enter an integer.",
        )
    else:
        num_processes = 1

    if distributed_type in [DistributedType.MULTI_GPU, DistributedType.NO] and not use_cpu and not use_mps:
        gpu_ids = _ask_field(
            "What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:",
            default="all",
        )

    if distributed_type == DistributedType.TPU:
        mixed_precision = "no"
        main_training_function = _ask_field(
            "What is the name of the function in your script that should be launched in all parallel scripts? [main]: ",
            default="main",
        )
        tpu_use_cluster = _ask_field(
            "Are you using a TPU cluster? [yes/NO]: ",
            _convert_yes_no_to_bool,
            default=False,
            error_message="Please enter yes or no.",
        )
        if tpu_use_cluster:
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
            tpu_use_sudo = _ask_field(
                "To run a python script in a TPU pod, should `sudo` be used? [yes/NO]: ",
                default=False,
                error_message="Please enter yes or no.",
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
                    tpu_command_file = _ask_field(
                        "What is the path to your bash script? ",
                        default=None,
                        error_message="Please enter the path to your bash script.",
                    )
                    tpu_command_file = os.path.abspath(tpu_command_file)
                else:
                    print("Please enter each command seperately you wish to run on startup in each pod.")
                    tpu_commands = []
                    another_command = True
                    while another_command:
                        tpu_commands.append(
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
            tpu_vm = _ask_field(
                "If not using an instance group, what are the names of the Compute VM instances to be used, seperated by a comma: ",
                default="",
            ).split(",")
            tpu_env = _ask_field(
                "What environment variables do you wish to set in each pod, seperated by a comma: ",
                default="",
            ).split(",")

    else:
        main_training_function = "main"
        if distributed_type == DistributedType.DEEPSPEED and use_deepspeed_config:
            mixed_precision = None
        else:
            mixed_precision = _ask_options(
                "Do you wish to use FP16 or BF16 (mixed precision)?",
                ["no", "fp16", "bf16", "fp8"],
                _convert_mixed_precision,
            )

    if use_dynamo and mixed_precision == "no" and not use_cpu:
        print(
            "Torch dynamo used without mixed precision requires TF32 to be efficient. Accelerate will enable it by default when launching your scripts."
        )

    if distributed_type == DistributedType.TPU and mixed_precision == "bf16":
        tpu_downcast_bf16 = _ask_field(
            "Should `torch.float` be cast as `bfloat16` and `torch.double` remain `float32` on TPUs?", default="no"
        )

    return ClusterConfig(
        compute_environment=ComputeEnvironment.LOCAL_MACHINE,
        distributed_type=distributed_type,
        num_processes=num_processes,
        gpu_ids=gpu_ids,
        mixed_precision=mixed_precision,
        downcast_bf16=tpu_downcast_bf16,
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
        commands=tpu_commands,
        command_file=tpu_command_file,
        tpu_env=tpu_env,
        tpu_name=tpu_name,
        tpu_vm=tpu_vm,
        tpu_zone=tpu_zone,
        tpu_use_sudo=tpu_use_sudo,
        tpu_use_cluster=tpu_use_cluster,
        dynamo_config=dynamo_config,
    )
