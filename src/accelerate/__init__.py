# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "0.16.0"

from .accelerator import Accelerator
from .big_modeling import (
    cpu_offload,
    disk_offload,
    dispatch_model,
    init_empty_weights,
    init_on_device,
    load_checkpoint_and_dispatch,
)
from .data_loader import skip_first_batches
from .launchers import debug_launcher, notebook_launcher
from .utils import (
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
