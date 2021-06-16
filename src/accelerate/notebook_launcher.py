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

from packaging import version

from .state import AcceleratorState
from .utils import PrepareForLaunch


def notebook_launcher(function, args=(), num_processes=None, use_fp16=False, use_port="29500"):
    """
    Launches a training function, using several processes if it's possible in the current environment (TPU with
    multiple cores for instance).

    Args:
        function (:obj:`Callable`):
            The training function to execute. If it accepts arguments, the first argument should be the index of the
            process run.
        args (:obj:`Tuple`):
            Tuple of arguments to pass to the function (it will receive :obj:`*args`).
        num_processes (:obj:`int`, `optional`):
            The number of processes to use for training. Will default to 8 in Colab/Kaggle if a TPU is available, to
            the number of GPUs available otherwise.
        use_fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, will use mixed precision training on multi-GPU.
        use_port (:obj:`str`, `optional`, defaults to :obj:`"29500"`):
            The port to use to communicate between processes when launching a multi-GPU training.
    """
    # Are we in a google colab or a Kaggle Kernel?
    if any(key.startswith("KAGGLE") for key in os.environ.keys()):
        in_colab_or_kaggle = True
    elif "IPython" in sys.modules:
        in_colab_or_kaggle = "google.colab" in str(sys.modules["IPython"].get_ipython())
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

            launcher = PrepareForLaunch(function, distributed_type="TPU")
            print(f"Launching a training on {num_processes} TPU cores.")
            xmp.spawn(launcher, args=args, nprocs=num_processes, start_method="fork")
        else:
            # No need for a distributed launch otherwise as it's either CPU or one GPU.
            if torch.cuda.is_available():
                print("Launching training on one GPU.")
            else:
                print("Launching training on CPU.")
            function(*args)

    else:
        if num_processes is None:
            raise ValueError(
                "You have to specify the number of GPUs you would like to use, add `num_process=...` to your call."
            )

        if num_processes > 1:
            # Multi-GPU launch
            if version.parse(torch.__version__) < version.parse("1.5.0"):
                raise ImportError(
                    "Using `notebook_launcher` for distributed training on GPUs require torch >= 1.5.0, got "
                    f"{torch.__version__}."
                )

            from torch.multiprocessing import start_processes

            if len(AcceleratorState._shared_state) > 0:
                raise ValueError(
                    "To launch a multi-GPU training from your notebook, the `Accelerator` should only be initialized "
                    "inside your training function. Restart your notebook and make sure no cells initializes an "
                    "`Accelerator`."
                )

            if torch.cuda.is_initialized():
                raise ValueError(
                    "To launch a multi-GPU training from your notebook, you need to avoid running any instruction "
                    "using `torch.cuda` in any cell. Restart your notebook and make sure no cells use any CUDA "
                    "function."
                )

            # torch.distributed will expect a few environment variable to be here. We set the ones common to each
            # process here (the other ones will be set be the launcher).
            os.environ["WORLD_SIZE"] = str(num_processes)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(use_port)
            os.environ["USE_FP16"] = str(use_fp16)

            launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")
            try:
                print(f"Launching a training on {num_processes} GPUs.")
                start_processes(launcher, args=args, nprocs=num_processes, start_method="fork")
            finally:
                # Clean up the environment variables set.
                del os.environ["WORLD_SIZE"]
                del os.environ["MASTER_ADDR"]
                del os.environ["MASTER_PORT"]

        else:
            # No need for a distributed launch otherwise as it's either CPU or one GPU.
            if torch.cuda.is_available():
                print("Launching training on one GPU.")
            else:
                print("Launching training on CPU.")
            function(*args)
