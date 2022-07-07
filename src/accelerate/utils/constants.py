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


SCALER_NAME = "scaler.pt"
MODEL_NAME = "pytorch_model"
RNG_STATE_NAME = "random_states"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
SAGEMAKER_PYTORCH_VERSION = "1.10.2"
SAGEMAKER_PYTHON_VERSION = "py38"
SAGEMAKER_TRANSFORMERS_VERSION = "4.17.0"
SAGEMAKER_PARALLEL_EC2_INSTANCES = ["ml.p3.16xlarge", "ml.p3dn.24xlarge", "ml.p4dn.24xlarge"]

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}
