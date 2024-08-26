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

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import yaml

from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION


hf_cache_home = os.path.expanduser(
    os.environ.get("HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_json_config_file = os.path.join(cache_dir, "default_config.yaml")
default_yaml_config_file = os.path.join(cache_dir, "default_config.yaml")

# For backward compatibility: the default config is the json one if it's the only existing file.
if os.path.isfile(default_yaml_config_file) or not os.path.isfile(default_json_config_file):
    default_config_file = default_yaml_config_file
else:
    default_config_file = default_json_config_file


def load_config_from_file(config_file):
    if config_file is not None:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"The passed configuration file `{config_file}` does not exist. "
                "Please pass an existing file to `accelerate launch`, or use the default one "
                "created through `accelerate config` and run `accelerate launch` "
                "without the `--config_file` argument."
            )
    else:
        config_file = default_config_file
    with open(config_file, encoding="utf-8") as f:
        if config_file.endswith(".json"):
            if (
                json.load(f).get("compute_environment", ComputeEnvironment.LOCAL_MACHINE)
                == ComputeEnvironment.LOCAL_MACHINE
            ):
                config_class = ClusterConfig
            else:
                config_class = SageMakerConfig
            return config_class.from_json_file(json_file=config_file)
        else:
            if (
                yaml.safe_load(f).get("compute_environment", ComputeEnvironment.LOCAL_MACHINE)
                == ComputeEnvironment.LOCAL_MACHINE
            ):
                config_class = ClusterConfig
            else:
                config_class = SageMakerConfig
            return config_class.from_yaml_file(yaml_file=config_file)


@dataclass
class BaseConfig:
    compute_environment: ComputeEnvironment
    distributed_type: Union[DistributedType, SageMakerDistributedType]
    mixed_precision: str
    use_cpu: bool
    debug: bool

    def to_dict(self):
        result = self.__dict__
        # For serialization, it's best to convert Enums to strings (or their underlying value type).

        def _convert_enums(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                if not bool(value):
                    return None
                for key1, value1 in value.items():
                    value[key1] = _convert_enums(value1)
            return value

        for key, value in result.items():
            result[key] = _convert_enums(value)
        result = {k: v for k, v in result.items() if v is not None}
        return result

    @staticmethod
    def process_config(config_dict):
        """
        Processes `config_dict` and sets default values for any missing keys
        """
        if "compute_environment" not in config_dict:
            config_dict["compute_environment"] = ComputeEnvironment.LOCAL_MACHINE
        if "distributed_type" not in config_dict:
            raise ValueError("A `distributed_type` must be specified in the config file.")
        if "num_processes" not in config_dict and config_dict["distributed_type"] == DistributedType.NO:
            config_dict["num_processes"] = 1
        if "mixed_precision" not in config_dict:
            config_dict["mixed_precision"] = "fp16" if ("fp16" in config_dict and config_dict["fp16"]) else None
        if "fp16" in config_dict:  # Convert the config to the new format.
            del config_dict["fp16"]
        if "dynamo_backend" in config_dict:  # Convert the config to the new format.
            dynamo_backend = config_dict.pop("dynamo_backend")
            config_dict["dynamo_config"] = {} if dynamo_backend == "NO" else {"dynamo_backend": dynamo_backend}
        if "use_cpu" not in config_dict:
            config_dict["use_cpu"] = False
        if "debug" not in config_dict:
            config_dict["debug"] = False
        if "enable_cpu_affinity" not in config_dict:
            config_dict["enable_cpu_affinity"] = False
        return config_dict

    @classmethod
    def from_json_file(cls, json_file=None):
        json_file = default_json_config_file if json_file is None else json_file
        with open(json_file, encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict = cls.process_config(config_dict)
        extra_keys = sorted(set(config_dict.keys()) - set(cls.__dataclass_fields__.keys()))
        if len(extra_keys) > 0:
            raise ValueError(
                f"The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `accelerate`"
                " version or fix (and potentially remove) these keys from your config file."
            )

        return cls(**config_dict)

    def to_json_file(self, json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            content = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
            f.write(content)

    @classmethod
    def from_yaml_file(cls, yaml_file=None):
        yaml_file = default_yaml_config_file if yaml_file is None else yaml_file
        with open(yaml_file, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict = cls.process_config(config_dict)
        extra_keys = sorted(set(config_dict.keys()) - set(cls.__dataclass_fields__.keys()))
        if len(extra_keys) > 0:
            raise ValueError(
                f"The config file at {yaml_file} had unknown keys ({extra_keys}), please try upgrading your `accelerate`"
                " version or fix (and potentially remove) these keys from your config file."
            )
        return cls(**config_dict)

    def to_yaml_file(self, yaml_file):
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)

    def __post_init__(self):
        if isinstance(self.compute_environment, str):
            self.compute_environment = ComputeEnvironment(self.compute_environment)
        if isinstance(self.distributed_type, str):
            if self.compute_environment == ComputeEnvironment.AMAZON_SAGEMAKER:
                self.distributed_type = SageMakerDistributedType(self.distributed_type)
            else:
                self.distributed_type = DistributedType(self.distributed_type)
        if getattr(self, "dynamo_config", None) is None:
            self.dynamo_config = {}


@dataclass
class ClusterConfig(BaseConfig):
    num_processes: int = -1  # For instance if we use SLURM and the user manually passes it in
    machine_rank: int = 0
    num_machines: int = 1
    gpu_ids: Optional[str] = None
    main_process_ip: Optional[str] = None
    main_process_port: Optional[int] = None
    rdzv_backend: Optional[str] = "static"
    same_network: Optional[bool] = False
    main_training_function: str = "main"
    enable_cpu_affinity: bool = False

    # args for FP8 training
    fp8_config: dict = None
    # args for deepspeed_plugin
    deepspeed_config: dict = None
    # args for fsdp
    fsdp_config: dict = None
    # args for megatron_lm
    megatron_lm_config: dict = None
    # args for ipex
    ipex_config: dict = None
    # args for mpirun
    mpirun_config: dict = None
    # args for TPU
    downcast_bf16: bool = False

    # args for TPU pods
    tpu_name: str = None
    tpu_zone: str = None
    tpu_use_cluster: bool = False
    tpu_use_sudo: bool = False
    command_file: str = None
    commands: List[str] = None
    tpu_vm: List[str] = None
    tpu_env: List[str] = None

    # args for dynamo
    dynamo_config: dict = None

    def __post_init__(self):
        if self.deepspeed_config is None:
            self.deepspeed_config = {}
        if self.fsdp_config is None:
            self.fsdp_config = {}
        if self.megatron_lm_config is None:
            self.megatron_lm_config = {}
        if self.ipex_config is None:
            self.ipex_config = {}
        if self.mpirun_config is None:
            self.mpirun_config = {}
        if self.fp8_config is None:
            self.fp8_config = {}
        return super().__post_init__()


@dataclass
class SageMakerConfig(BaseConfig):
    ec2_instance_type: str
    iam_role_name: str
    image_uri: Optional[str] = None
    profile: Optional[str] = None
    region: str = "us-east-1"
    num_machines: int = 1
    gpu_ids: str = "all"
    base_job_name: str = f"accelerate-sagemaker-{num_machines}"
    pytorch_version: str = SAGEMAKER_PYTORCH_VERSION
    transformers_version: str = SAGEMAKER_TRANSFORMERS_VERSION
    py_version: str = SAGEMAKER_PYTHON_VERSION
    sagemaker_inputs_file: str = None
    sagemaker_metrics_file: str = None
    additional_args: dict = None
    dynamo_config: dict = None
    enable_cpu_affinity: bool = False
