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
import tempfile

import torch

from .state import AcceleratorState
from .utils import PrecisionType, PrepareForLaunch, patch_environment


def notebook_launcher(function, args=(), num_processes=None, mixed_precision="no", use_port="29500"):
    """
    Launches a training function, using several processes if it's possible in the current environment (TPU with
    multiple cores for instance).

    <Tip warning={true}>

    To use this function absolutely zero calls to a CUDA device must be made in the notebook session before calling. If
    any have been made, you will need to restart the notebook and make sure no cells use any CUDA capability.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute. If it accepts arguments, the first argument should be the index of the
            process run.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*):
            The number of processes to use for training. Will default to 8 in Colab/Kaggle if a TPU is available, to
            the number of GPUs available otherwise.
        mixed_precision (`str`, *optional*, defaults to `"no"`):
            If `fp16` or `bf16`, will use mixed precision training on multi-GPU.
        use_port (`str`, *optional*, defaults to `"29500"`):
            The port to use to communicate between processes when launching a multi-GPU training.

    Example:

    ```python
    # Assume this is defined in a Jupyter Notebook on an instance with two GPUs
    from accelerate import notebook_launcher


    def train(*args):
        # Your training function here
        ...


    notebook_launcher(train, args=(arg1, arg2), num_processes=2, mixed_precision="fp16")
    ```
    """
    # Are we in a google colab or a Kaggle Kernel?
    in_colab = False
    in_kaggle = False
    if any(key.startswith("KAGGLE") for key in os.environ.keys()):
        in_kaggle = True
    elif "IPython" in sys.modules:
        in_colab = "google.colab" in str(sys.modules["IPython"].get_ipython())

    try:
        mixed_precision = PrecisionType(mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if (in_colab or in_kaggle) and (os.environ.get("TPU_NAME", None) is not None):
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
    elif in_colab:
        # No need for a distributed launch otherwise as it's either CPU or one GPU.
        if torch.cuda.is_available():
            print("Launching training on one GPU.")
        else:
            print("Launching training on one CPU.")
        function(*args)
    else:
        if num_processes is None:
            raise ValueError(
                "You have to specify the number of GPUs you would like to use, add `num_processes=...` to your call."
            )

        if num_processes > 1:
            # Multi-GPU launch
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
            with patch_environment(
                world_size=num_processes, master_addr="127.0.01", master_port=use_port, mixed_precision=mixed_precision
            ):
                launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")

                print(f"Launching training on {num_processes} GPUs.")
                start_processes(launcher, args=args, nprocs=num_processes, start_method="fork")

        else:
            # No need for a distributed launch otherwise as it's either CPU, GPU or MPS.
            use_mps_device = "false"
            if torch.backends.mps.is_available():
                print("Launching training on MPS.")
                use_mps_device = "true"
            elif torch.cuda.is_available():
                print("Launching training on one GPU.")
            else:
                print("Launching training on CPU.")
            with patch_environment(accelerate_use_mps_device=use_mps_device):
                function(*args)


def debug_launcher(function, args=(), num_processes=2):
    """
    Launches a training function using several processes on CPU for debugging purposes.

    <Tip warning={true}>

    This function is provided for internal testing and debugging, but it's not intended for real trainings. It will
    only use the CPU.

    </Tip>

    Args:
        function (`Callable`):
            The training function to execute.
        args (`Tuple`):
            Tuple of arguments to pass to the function (it will receive `*args`).
        num_processes (`int`, *optional*, defaults to 2):
            The number of processes to use for training.
    """
    from torch.multiprocessing import start_processes

    with tempfile.NamedTemporaryFile() as tmp_file:
        # torch.distributed will expect a few environment variable to be here. We set the ones common to each
        # process here (the other ones will be set be the launcher).
        with patch_environment(
            world_size=num_processes,
            master_addr="127.0.01",
            master_port="29500",
            accelerate_mixed_precision="no",
            accelerate_debug_rdv_file=tmp_file.name,
            accelerate_use_cpu="yes",
        ):
            launcher = PrepareForLaunch(function, debug=True)
            start_processes(launcher, args=args, nprocs=num_processes, start_method="fork")
