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

import operator as op

import torch


SCALER_NAME = "scaler.pt"
MODEL_NAME = "pytorch_model"
SAFE_MODEL_NAME = "model"
RNG_STATE_NAME = "random_states"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
SAMPLER_NAME = "sampler"
PROFILE_PATTERN_NAME = "profile_{suffix}.json"
WEIGHTS_NAME = f"{MODEL_NAME}.bin"
WEIGHTS_PATTERN_NAME = "pytorch_model{suffix}.bin"
WEIGHTS_INDEX_NAME = f"{WEIGHTS_NAME}.index.json"
SAFE_WEIGHTS_NAME = f"{SAFE_MODEL_NAME}.safetensors"
SAFE_WEIGHTS_PATTERN_NAME = "model{suffix}.safetensors"
SAFE_WEIGHTS_INDEX_NAME = f"{SAFE_WEIGHTS_NAME}.index.json"
SAGEMAKER_PYTORCH_VERSION = "1.10.2"
SAGEMAKER_PYTHON_VERSION = "py38"
SAGEMAKER_TRANSFORMERS_VERSION = "4.17.0"
SAGEMAKER_PARALLEL_EC2_INSTANCES = ["ml.p3.16xlarge", "ml.p3dn.24xlarge", "ml.p4dn.24xlarge"]
FSDP_SHARDING_STRATEGY = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "HYBRID_SHARD_ZERO2"]
FSDP_AUTO_WRAP_POLICY = ["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP", "NO_WRAP"]
FSDP_BACKWARD_PREFETCH = ["BACKWARD_PRE", "BACKWARD_POST", "NO_PREFETCH"]
FSDP_STATE_DICT_TYPE = ["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"]
FSDP2_STATE_DICT_TYPE = ["SHARDED_STATE_DICT", "FULL_STATE_DICT"]
FSDP_PYTORCH_VERSION = (
    "2.1.0.a0+32f93b1"  # Technically should be 2.1.0, but MS-AMP uses this specific prerelease in their Docker image.
)
FSDP2_PYTORCH_VERSION = "2.6.0"
FSDP_MODEL_NAME = "pytorch_model_fsdp"
DEEPSPEED_MULTINODE_LAUNCHERS = ["pdsh", "standard", "openmpi", "mvapich", "mpich", "nossh", "slurm"]
TORCH_DYNAMO_MODES = ["default", "reduce-overhead", "max-autotune"]
ELASTIC_LOG_LINE_PREFIX_TEMPLATE_PYTORCH_VERSION = "2.2.0"
XPU_PROFILING_AVAILABLE_PYTORCH_VERSION = "2.4.0"
MITA_PROFILING_AVAILABLE_PYTORCH_VERSION = "2.1.0"
BETA_TP_AVAILABLE_PYTORCH_VERSION = "2.3.0"

BETA_TP_AVAILABLE_TRANSFORMERS_VERSION = "4.52.0"
BETA_CP_AVAILABLE_PYTORCH_VERSION = "2.6.0"

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

# These are the args for `torch.distributed.launch` for pytorch < 1.9
TORCH_LAUNCH_PARAMS = [
    "nnodes",
    "nproc_per_node",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "rdzv_conf",
    "standalone",
    "max_restarts",
    "monitor_interval",
    "start_method",
    "role",
    "module",
    "m",
    "no_python",
    "run_path",
    "log_dir",
    "r",
    "redirects",
    "t",
    "tee",
    "node_rank",
    "master_addr",
    "master_port",
]

CUDA_DISTRIBUTED_TYPES = ["DEEPSPEED", "MULTI_GPU", "FSDP", "MEGATRON_LM", "TP"]
TORCH_DISTRIBUTED_OPERATION_TYPES = CUDA_DISTRIBUTED_TYPES + [
    "MULTI_NPU",
    "MULTI_MLU",
    "MULTI_SDAA",
    "MULTI_MUSA",
    "MULTI_XPU",
    "MULTI_CPU",
    "MULTI_HPU",
]
SUPPORTED_PYTORCH_LAYERS_FOR_UPCASTING = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
)
