# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "0.8.0.dev0"

from .accelerator import Accelerator
from .launchers import debug_launcher, notebook_launcher
from .utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    synchronize_rng_states,
)
