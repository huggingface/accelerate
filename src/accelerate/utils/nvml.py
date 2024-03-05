# Copyright 2024 The HuggingFace Team and NVIDIA. All rights reserved.
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

import math
import os
from functools import lru_cache
from typing import List

from .imports import is_pynvml_available


if is_pynvml_available():
    import pynvml


@lru_cache
def init_pynvml():
    "Init and cache pynvml only when needed"
    pynvml.nvmlInit()

# assume nvml returns list of 64 bit ints


nvml_affinity_elements = math.ceil(os.cpu_count() / 64)


def get_cpu_affinity(device_index: int = 0) -> List[int]:
    """
    Gets the CPU affinity for a given GPU index
    """
    if not is_pynvml_available():
        raise ImportError("Setting the CPU affinity requires the `pynvml` library to be installed. Please run `pip install pynvml`")
    init_pynvml()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    affinity_string = ""
    for i in pynvml.nvmlDeviceGetCpuAffinity(
        handle, nvml_affinity_elements
    ):
        # Assume nvml returns a list of 64 bit integers
        affinity_string = f"{i:064b}{affinity_string}"
    affinity_list = [int(x) for x in affinity_string]
    affinity_list.reverse()  # To make the core 0 the first element
    return [i for i, e in enumerate(affinity_list) if e != 0]


def set_cpu_affinity(device_index: int = None):
    """
    Sets the CPU affinity for a given GPU index based on the
    result of `get_cpu_affinity`
    """
    if not is_pynvml_available():
        raise ImportError("Setting the CPU affinity requires the `pynvml` library to be installed. Please run `pip install pynvml`")
    init_pynvml()
    if device_index is None:
        device_index = int(os.environ.get("LOCAL_RANK", 0))
    os.sched_setaffinity(0, get_cpu_affinity(device_index))
    return os.sched_getaffinity(0)
