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
__version__ = "0.31.0"

from .accelerator import Accelerator
from .big_modeling import (
    cpu_offload,
    cpu_offload_with_hook,
    disk_offload,
    dispatch_model,
    init_empty_weights,
    init_on_device,
    load_checkpoint_and_dispatch,
)
from .data_loader import skip_first_batches
from .inference import prepare_pippy
from .launchers import debug_launcher, notebook_launcher
from .state import PartialState
from .utils import (
    AutocastKwargs,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    FullyShardedDataParallelPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    find_executable_batch_size,
    infer_auto_device_map,
    is_rich_available,
    load_checkpoint_in_model,
    synchronize_rng_states,
)


if is_rich_available():
    from .utils import rich
