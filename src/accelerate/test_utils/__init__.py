# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from .testing import (
    DEFAULT_LAUNCH_COMMAND,
    are_the_same_tensors,
    assert_exception,
    capture_call_output,
    device_count,
    execute_subprocess_async,
    get_launch_command,
    memory_allocated_func,
    path_in_accelerate_package,
    require_bnb,
    require_cpu,
    require_cuda,
    require_huggingface_suite,
    require_mlu,
    require_mps,
    require_multi_device,
    require_multi_gpu,
    require_multi_xpu,
    require_musa,
    require_non_cpu,
    require_non_torch_xla,
    require_non_xpu,
    require_npu,
    require_pippy,
    require_single_device,
    require_single_gpu,
    require_single_xpu,
    require_torch_min_version,
    require_torchvision,
    require_tpu,
    require_transformer_engine,
    require_xpu,
    skip,
    slow,
    torch_device,
)
from .training import RegressionDataset, RegressionModel, RegressionModel4XPU


from .scripts import test_script, test_sync, test_ops  # isort: skip
