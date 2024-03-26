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

from ...utils.dataclasses import (
    ComputeEnvironment,
    DistributedType,
    DynamoBackend,
    PrecisionType,
    SageMakerDistributedType,
)
from ..menu import BulletMenu


DYNAMO_BACKENDS = [
    "EAGER",
    "AOT_EAGER",
    "INDUCTOR",
    "AOT_TS_NVFUSER",
    "NVPRIMS_NVFUSER",
    "CUDAGRAPHS",
    "OFI",
    "FX2TRT",
    "ONNXRT",
    "TENSORRT",
    "IPEX",
    "TVM",
]


def _ask_field(input_text, convert_value=None, default=None, error_message=None):
    ask_again = True
    while ask_again:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result) if convert_value is not None else result
        except Exception:
            if error_message is not None:
                print(error_message)


def _ask_options(input_text, options=[], convert_value=None, default=0):
    menu = BulletMenu(input_text, options)
    result = menu.run(default_choice=default)
    return convert_value(result) if convert_value is not None else result


def _convert_compute_environment(value):
    value = int(value)
    return ComputeEnvironment(["LOCAL_MACHINE", "AMAZON_SAGEMAKER"][value])


def _convert_distributed_mode(value):
    value = int(value)
    return DistributedType(["NO", "MULTI_CPU", "MULTI_XPU", "MULTI_GPU", "MULTI_NPU", "MULTI_MLU", "XLA"][value])


def _convert_dynamo_backend(value):
    value = int(value)
    return DynamoBackend(DYNAMO_BACKENDS[value]).value


def _convert_mixed_precision(value):
    value = int(value)
    return PrecisionType(["no", "fp16", "bf16", "fp8"][value])


def _convert_sagemaker_distributed_mode(value):
    value = int(value)
    return SageMakerDistributedType(["NO", "DATA_PARALLEL", "MODEL_PARALLEL"][value])


def _convert_yes_no_to_bool(value):
    return {"yes": True, "no": False}[value.lower()]


class SubcommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    A custom formatter that will remove the usage line from the help message for subcommands.
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage = super()._format_usage(usage, actions, groups, prefix)
        usage = usage.replace("<command> [<args>] ", "")
        return usage
