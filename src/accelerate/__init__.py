# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "0.4.0"

from .accelerator import Accelerator
from .kwargs_handlers import DistributedDataParallelKwargs, GradScalerKwargs
from .notebook_launcher import notebook_launcher
from .state import DistributedType
from .utils import DeepSpeedPlugin, synchronize_rng_states
