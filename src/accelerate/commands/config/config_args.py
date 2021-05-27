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
from typing import Optional, Union

import yaml
from accelerate.state import ComputeEnvironment, DistributedType, SageMakerDistributedType


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_json_config_file = os.path.join(cache_dir, "default_config.json")
default_yaml_config_file = os.path.join(cache_dir, "default_config.yaml")

# For backward compatibility: the default config is the json one if it's the only existing file.
if os.path.isfile(default_yaml_config_file) or not os.path.isfile(default_json_config_file):
    default_config_file = default_yaml_config_file
else:
    default_config_file = default_json_config_file


def load_config_from_file(config_file):
    config_file_exists = config_file is not None and os.path.isfile(config_file)
    config_file = config_file if config_file_exists else default_config_file
    with open(config_file, "r", encoding="utf-8") as f:
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
    fp16: bool

    def to_dict(self):
        result = self.__dict__
        # For serialization, it's best to convert Enums to strings (or their underlying value type).
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        return result

    @classmethod
    def from_json_file(cls, json_file=None):
        json_file = default_json_config_file if json_file is None else json_file
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if "compute_environment" not in config_dict:
            config_dict["compute_environment"] = ComputeEnvironment.LOCAL_MACHINE
        return cls(**config_dict)

    def to_json_file(self, json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            content = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
            f.write(content)

    @classmethod
    def from_yaml_file(cls, yaml_file=None):
        yaml_file = default_yaml_config_file if yaml_file is None else yaml_file
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        if "compute_environment" not in config_dict:
            config_dict["compute_environment"] = ComputeEnvironment.LOCAL_MACHINE
        return cls(**config_dict)

    def to_yaml_file(self, yaml_file):
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)

    def __post_init__(self):
        if isinstance(self.compute_environment, str):
            self.compute_environment = ComputeEnvironment(self.compute_environment)
        if isinstance(self.distributed_type, str):
            self.distributed_type = DistributedType(self.distributed_type)


@dataclass
class ClusterConfig(BaseConfig):
    num_processes: int
    machine_rank: int = 0
    num_machines: int = 1
    main_process_ip: Optional[str] = None
    main_process_port: Optional[int] = None
    main_training_function: str = "main"

    # args for deepspeed_plugin
    deepspeed_config: dict = None


@dataclass
class SageMakerConfig(BaseConfig):
    ec2_instance_type: str
    iam_role_name: str
    profile: Optional[str] = None
    region: str = "us-east-1"
    num_machines: int = 1
    base_job_name: str = f"accelerate-sagemaker-{num_machines}"
    pytorch_version: str = "1.6"
    transformers_version: str = "4.4"
