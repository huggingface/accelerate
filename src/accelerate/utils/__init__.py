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
from .constants import (
    MITA_PROFILING_AVAILABLE_PYTORCH_VERSION,
    MODEL_NAME,
    OPTIMIZER_NAME,
    PROFILE_PATTERN_NAME,
    RNG_STATE_NAME,
    SAFE_MODEL_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_PATTERN_NAME,
    SAMPLER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TORCH_DISTRIBUTED_OPERATION_TYPES,
    TORCH_LAUNCH_PARAMS,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_PATTERN_NAME,
    XPU_PROFILING_AVAILABLE_PYTORCH_VERSION,
)
from .dataclasses import (
    AutocastKwargs,
    BnbQuantizationConfig,
    ComputeEnvironment,
    CustomDtype,
    DataLoaderConfiguration,
    DDPCommunicationHookType,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    DynamoBackend,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProfileKwargs,
    ProjectConfiguration,
    RNGType,
    SageMakerDistributedType,
    TensorInformation,
    TorchDynamoPlugin,
    add_model_config_to_megatron_parser,
)
from .environment import (
    are_libraries_initialized,
    check_cuda_p2p_ib_support,
    check_fp8_capability,
    clear_environment,
    convert_dict_to_env_variables,
    get_cpu_distributed_information,
    get_gpu_info,
    get_int_from_env,
    parse_choice_from_env,
    parse_flag_from_env,
    patch_environment,
    set_numa_affinity,
    str_to_bool,
)
from .imports import (
    deepspeed_required,
    get_ccl_version,
    is_4bit_bnb_available,
    is_8bit_bnb_available,
    is_aim_available,
    is_bf16_available,
    is_bitsandbytes_multi_backend_available,
    is_bnb_available,
    is_boto3_available,
    is_ccl_available,
    is_clearml_available,
    is_comet_ml_available,
    is_cuda_available,
    is_datasets_available,
    is_deepspeed_available,
    is_dvclive_available,
    is_fp8_available,
    is_import_timer_available,
    is_ipex_available,
    is_lomo_available,
    is_megatron_lm_available,
    is_mlflow_available,
    is_mlu_available,
    is_mps_available,
    is_msamp_available,
    is_musa_available,
    is_npu_available,
    is_pandas_available,
    is_peft_available,
    is_pippy_available,
    is_pynvml_available,
    is_pytest_available,
    is_rich_available,
    is_sagemaker_available,
    is_schedulefree_available,
    is_tensorboard_available,
    is_timm_available,
    is_torch_xla_available,
    is_torchdata_available,
    is_torchdata_stateful_dataloader_available,
    is_torchvision_available,
    is_transformer_engine_available,
    is_transformers_available,
    is_triton_available,
    is_wandb_available,
    is_weights_only_available,
    is_xpu_available,
)
from .modeling import (
    align_module_device,
    calculate_maximum_sizes,
    check_device_map,
    check_tied_parameters_in_config,
    check_tied_parameters_on_same_device,
    compute_module_sizes,
    convert_file_size_to_int,
    dtype_byte_size,
    find_tied_parameters,
    get_balanced_memory,
    get_grad_scaler,
    get_max_layer_size,
    get_max_memory,
    get_mixed_precision_context_manager,
    has_offloaded_params,
    id_tensor_storage,
    infer_auto_device_map,
    is_peft_model,
    load_checkpoint_in_model,
    load_offloaded_weights,
    load_state_dict,
    named_module_tensors,
    retie_parameters,
    set_module_tensor_to_device,
)
from .offload import (
    OffloadedWeightsLoader,
    PrefixedDataset,
    extract_submodules_state_dict,
    load_offloaded_weight,
    offload_state_dict,
    offload_weight,
    save_offload_index,
)
from .operations import (
    CannotPadNestedTensorWarning,
    GatheredParameters,
    broadcast,
    broadcast_object_list,
    concatenate,
    convert_outputs_to_fp32,
    convert_to_fp32,
    copy_tensor_to_devices,
    find_batch_size,
    find_device,
    gather,
    gather_object,
    get_data_structure,
    honor_type,
    ignorant_find_batch_size,
    initialize_tensors,
    is_namedtuple,
    is_tensor_information,
    is_torch_tensor,
    listify,
    pad_across_processes,
    pad_input_tensors,
    recursively_apply,
    reduce,
    send_to_device,
    slice_tensors,
)
from .versions import compare_versions, is_torch_version


if is_deepspeed_available():
    from .deepspeed import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
        HfDeepSpeedConfig,
        get_active_deepspeed_plugin,
    )

from .bnb import has_4bit_bnb_layers, load_and_quantize_model
from .fsdp_utils import (
    disable_fsdp_ram_efficient_loading,
    enable_fsdp_ram_efficient_loading,
    ensure_weights_retied,
    load_fsdp_model,
    load_fsdp_optimizer,
    merge_fsdp_weights,
    save_fsdp_model,
    save_fsdp_optimizer,
)
from .launch import (
    PrepareForLaunch,
    _filter_args,
    prepare_deepspeed_cmd_env,
    prepare_multi_gpu_env,
    prepare_sagemager_args_inputs,
    prepare_simple_launcher_cmd_env,
    prepare_tpu,
)

# For docs
from .megatron_lm import (
    AbstractTrainStep,
    BertTrainStep,
    GPTTrainStep,
    MegatronLMDummyDataLoader,
    MegatronLMDummyScheduler,
    T5TrainStep,
    avg_losses_across_data_parallel_group,
)


if is_megatron_lm_available():
    from .megatron_lm import (
        MegatronEngine,
        MegatronLMOptimizerWrapper,
        MegatronLMSchedulerWrapper,
        gather_across_data_parallel_groups,
    )
    from .megatron_lm import initialize as megatron_lm_initialize
    from .megatron_lm import prepare_data_loader as megatron_lm_prepare_data_loader
    from .megatron_lm import prepare_model_optimizer_scheduler as megatron_lm_prepare_model_optimizer_scheduler
    from .megatron_lm import prepare_optimizer as megatron_lm_prepare_optimizer
    from .megatron_lm import prepare_scheduler as megatron_lm_prepare_scheduler
from .memory import find_executable_batch_size, release_memory
from .other import (
    check_os_kernel,
    clean_state_dict_for_safetensors,
    convert_bytes,
    extract_model_from_parallel,
    get_pretty_name,
    is_port_in_use,
    load,
    merge_dicts,
    recursive_getattr,
    save,
    wait_for_everyone,
    write_basic_config,
)
from .random import set_seed, synchronize_rng_state, synchronize_rng_states
from .torch_xla import install_xla
from .tqdm import tqdm
from .transformer_engine import (
    apply_fp8_autowrap,
    contextual_fp8_autocast,
    convert_model,
    has_transformer_engine_layers,
)
