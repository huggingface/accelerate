# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all

from .constants import MODEL_NAME, OPTIMIZER_NAME, RNG_STATE_NAME, SCALER_NAME, SCHEDULER_NAME
from .dataclasses import (
    ComputeEnvironment,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    FullyShardedDataParallelPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    PrecisionType,
    RNGType,
    SageMakerDistributedType,
    TensorInformation,
)
from .imports import (
    is_apex_available,
    is_boto3_available,
    is_ccl_available,
    is_comet_ml_available,
    is_deepspeed_available,
    is_sagemaker_available,
    is_tensorboard_available,
    is_tensorflow_available,
    is_tpu_available,
    is_wandb_available,
)
from .modeling import (
    check_device_map,
    compute_module_sizes,
    convert_file_size_to_int,
    dtype_byte_size,
    find_tied_parameters,
    get_max_layer_size,
    get_max_memory,
    infer_auto_device_map,
    load_checkpoint_in_model,
    load_offloaded_weights,
    named_module_tensors,
    set_module_tensor_to_device,
)
from .offload import (
    OffloadedWeightsLoader,
    PrefixedDataset,
    extract_submodules_state_dict,
    offload_state_dict,
    offload_weight,
    save_offload_index,
)
from .operations import (
    broadcast,
    broadcast_object_list,
    concatenate,
    convert_outputs_to_fp32,
    convert_to_fp32,
    find_batch_size,
    find_device,
    gather,
    gather_object,
    get_data_structure,
    honor_type,
    initialize_tensors,
    is_tensor_information,
    is_torch_tensor,
    pad_across_processes,
    recursively_apply,
    reduce,
    send_to_device,
    slice_tensors,
)


if is_deepspeed_available():
    from .deepspeed import DeepSpeedEngineWrapper, DeepSpeedOptimizerWrapper

from .launch import PrepareForLaunch
from .memory import find_executable_batch_size
from .other import (
    extract_model_from_parallel,
    get_pretty_name,
    patch_environment,
    save,
    wait_for_everyone,
    write_basic_config,
)
from .random import set_seed, synchronize_rng_state, synchronize_rng_states
