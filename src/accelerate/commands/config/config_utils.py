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

from accelerate.state import ComputeEnvironment, DistributedType, SageMakerDistributedType


def _ask_field(input_text, convert_value=None, default=None, error_message=None):
    ask_again = True
    while ask_again:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result) if convert_value is not None else result
        except:
            if error_message is not None:
                print(error_message)


def _convert_compute_environment(value):
    value = int(value)
    return ComputeEnvironment(["LOCAL_MACHINE", "AMAZON_SAGEMAKER"][value])


def _convert_distributed_mode(value):
    value = int(value)
    return DistributedType(["NO", "MULTI_CPU", "MULTI_GPU", "TPU"][value])


def _convert_sagemaker_distributed_mode(value):
    value = int(value)
    return SageMakerDistributedType(["NO", "DATA_PARALLEL", "MODEL_PARALLEL"][value])


def _convert_yes_no_to_bool(value):
    return {"yes": True, "no": False}[value.lower()]
