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
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yaml
from accelerate.state import DistributedType, ComputeEnvironment

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


@dataclass
class BaseConfig:
    compute_environment: ComputeEnvironment

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
            return cls(**json.load(f))

    def to_json_file(self, json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            content = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
            f.write(content)

    @classmethod
    def from_yaml_file(cls, yaml_file=None):
        yaml_file = default_yaml_config_file if yaml_file is None else yaml_file
        with open(yaml_file, "r", encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml_file(self, yaml_file):
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)

    def __post_init__(self):
        if isinstance(self.distributed_type, str):
            self.compute_environment = ComputeEnvironment(self.compute_environment)


@dataclass
class ClusterConfig(BaseConfig):
    distributed_type: DistributedType
    num_processes: int
    fp16: bool
    machine_rank: int = 0
    num_machines: int = 1
    main_process_ip: Optional[str] = None
    main_process_port: Optional[int] = None
    main_training_function: str = "main"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.distributed_type, str):
            self.distributed_type = DistributedType(self.distributed_type)
