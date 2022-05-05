# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "0.8.0.dev0"

from .accelerator import Accelerator
from .big_modeling import cpu_offload, disk_offload, dispatch_model, init_empty_weights
from .launchers import debug_launcher, notebook_launcher
from .utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    find_executable_batch_size,
    synchronize_rng_states,
)
