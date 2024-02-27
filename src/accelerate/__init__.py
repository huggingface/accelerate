__version__ = "0.25.0.dev0"

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
from .launchers import debug_launcher, notebook_launcher
from .state import PartialState
from .utils import (
    AutocastKwargs,
    AxoNNPlugin,
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
