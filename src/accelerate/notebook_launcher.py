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
import warnings

import torch

from .state import AcceleratorState
from .utils import PrepareForLaunch


def notebook_launcher(function, args=(), num_processes=None, start_method="fork", **kwargs):
    """
    Launches a training function, using several processes if it's possible in the current environment (TPU with
    multiple cores for instance).

    Args:
        function (:obj:`Callable`):
            The training function to execute. If it accepts arguments, the first argument should be the index of the
            process run.
        args (:obj:`Tuple`):
            Tuple of arguments to pass to the function (it will receive :obj:`(index, *args)`).
        num_processes (:obj:`int`, `optional`):
            The number of processes to use for training. Will default to 8 in Colab/Kaggle if a TPU is available, to
            the number of GPUs available otherwise.

    .. warning::

        Multiple GPUs is not yet supported.
    """
    launcher = PrepareForLaunch(function)

    # Are we in a google colab or a Kaggle Kernel?
    if "IPython" in sys.modules:
        in_colab_or_kaggle = "google.colab" in str(sys.modules["IPython"].get_ipython())
    elif any(key.startswith("KAGGLE") for key in os.environ.keys()):
        in_colab_or_kaggle = True
    else:
        in_colab_or_kaggle = False

    if in_colab_or_kaggle:
        if os.environ.get("TPU_NAME", None) is not None:
            # TPU launch
            import torch_xla.distributed.xla_multiprocessing as xmp

            if len(AcceleratorState._shared_state) > 0:
                raise ValueError(
                    "To train on TPU in Colab or Kaggle Kernel, the `Accelerator` should only be initialized inside "
                    "your training function. Restart your notebook and make sure no cells initializes an "
                    "`Accelerator`."
                )
            if num_processes is None:
                num_processes = 8

            xmp.spawn(launcher, args=args, nprocs=num_processes, start_method="fork")
        else:
            # No need for a distributed launch otherwise as it's either CPU or one GPU.
            launcher(0, *args)

    else:
        if num_processes is None:
            num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if num_processes > 1:
            warnings.warn("`notebook_launcher` does not support multiple GPUs yet, launching the training on one GPU.")
        launcher(0, *args)
