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

import os
import sys

import torch

from .state import AcceleratorState
from .utils import PrepareForLaunch


def notebook_launcher(function, args=(), num_processes=None, **kwargs):
    """
    Launches a training
    """
    launcher = PrepareForLaunch(function)

    # Are we in a google colab?
    if "IPython" in sys.modules:
        in_colab = "google.colab" in str(sys.modules["IPython"].get_ipython())
    else:
        in_colab = False

    if in_colab:
        if os.environ.get("TPU_NAME", None) is not None:
            # TPU launch
            import torch_xla.distributed.xla_multiprocessing as xmp

            if len(AcceleratorState._shared_state) > 0:
                raise ValueError(
                    "To train on TPU in colab, the `Accelerator` should only be initialized inside your training "
                    "function. Restart your notebook and make sure no cells initializes an `Accelerator`."
                )
            if num_processes is None:
                num_processes = 8

            xmp.spawn(launcher, args=args, nprocs=num_processes, **kwargs)
        else:
            # No need for a distributed launch otherwise as it's either CPU or one GPU.
            launcher(0, *args)

    else:
        if num_processes is None:
            num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_processes > 1:
            torch.multiprocessing.spwan(launcher, args=args, nprocs=num_processes, **kwargs)
        else:
            # No need for a distributed launch otherwise as it's either CPU or one GPU.
            launcher(0, *args)
