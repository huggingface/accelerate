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

from .state import AcceleratorState, PartialState
from .utils import (
    PrecisionType,
    PrepareForLaunch,
    are_libraries_initialized,
    check_cuda_p2p_ib_support,
    get_gpu_info,
    is_mps_available,
    is_torch_version,
    patch_environment,
)
from .utils.constants import ELASTIC_LOG_LINE_PREFIX_TEMPLATE_PYTORCH_VERSION


def test_launch():
    "Verify a `PartialState` can be initialized."
    _ = PartialState()


def notebook_launcher(
    function,
    args=(),
    num_processes=None,
    mixed_precision="no",
    use_port="29500",
    master_addr="127.0.0.1",
    node_rank=0,
    num_nodes=1,
    rdzv_backend="static",
    rdzv_endpoint="",
    rdzv_conf=None,
    rdzv_id="none",
    max_restarts=0,
    monitor_interval=0.1,
    log_line_prefix_template=None,
):
    """
    Launches a training function, using several processes or multiple nodes if it's possible in the current environment
    (TPU with multiple cores for instance).

    <Tip warning={true}>

    To use this function absolutely zero calls to a CUDA device must be made in the notebook session before calling. If
    any have been made, you will need to restart the notebook and make sure no cells use any CUDA capability.

    Setting `ACCELERATE_DEBUG_MODE="1"` in your environment will run a test before truly launching to ensure that none
    of those calls have been made.

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
        master_addr (`str`, *optional*, defaults to `"127.0.0.1"`):
            The address to use for communication between processes.
        node_rank (`int`, *optional*, defaults to 0):
            The rank of the current node.
        num_nodes (`int`, *optional*, defaults to 1):
            The number of nodes to use for training.
        rdzv_backend (`str`, *optional*, defaults to `"static"`):
            The rendezvous method to use, such as 'static' (the default) or 'c10d'
        rdzv_endpoint (`str`, *optional*, defaults to `""`):
            The endpoint of the rdzv sync. storage.
        rdzv_conf (`Dict`, *optional*, defaults to `None`):
            Additional rendezvous configuration.
        rdzv_id (`str`, *optional*, defaults to `"none"`):
            The unique run id of the job.
        max_restarts (`int`, *optional*, defaults to 0):
            The maximum amount of restarts that elastic agent will conduct on workers before failure.
        monitor_interval (`float`, *optional*, defaults to 0.1):
            The interval in seconds that is used by the elastic_agent as a period of monitoring workers.
        log_line_prefix_template (`str`, *optional*, defaults to `None`):
            The prefix template for elastic launch logging. Available from PyTorch 2.2.0.

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

        launcher = PrepareForLaunch(function, distributed_type="XLA")
        print(f"Launching a training on {num_processes} TPU cores.")
        xmp.spawn(launcher, args=args, nprocs=num_processes, start_method="fork")
    elif in_colab and get_gpu_info()[1] < 2:
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
        if node_rank >= num_nodes:
            raise ValueError("The node_rank must be less than the number of nodes.")
        if num_processes > 1:
            # Multi-GPU launch
            from torch.distributed.launcher.api import LaunchConfig, elastic_launch
            from torch.multiprocessing import start_processes
            from torch.multiprocessing.spawn import ProcessRaisedException

            if len(AcceleratorState._shared_state) > 0:
                raise ValueError(
                    "To launch a multi-GPU training from your notebook, the `Accelerator` should only be initialized "
                    "inside your training function. Restart your notebook and make sure no cells initializes an "
                    "`Accelerator`."
                )
            # Check for specific libraries known to initialize CUDA that users constantly use
            problematic_imports = are_libraries_initialized("bitsandbytes")
            if len(problematic_imports) > 0:
                err = (
                    "Could not start distributed process. Libraries known to initialize CUDA upon import have been "
                    "imported already. Please keep these imports inside your training function to try and help with this:"
                )
                for lib_name in problematic_imports:
                    err += f"\n\t* `{lib_name}`"
                raise RuntimeError(err)

            patched_env = dict(
                nproc=num_processes,
                node_rank=node_rank,
                world_size=num_nodes * num_processes,
                master_addr=master_addr,
                master_port=use_port,
                mixed_precision=mixed_precision,
            )

            # Check for CUDA P2P and IB issues
            if not check_cuda_p2p_ib_support():
                patched_env["nccl_p2p_disable"] = "1"
                patched_env["nccl_ib_disable"] = "1"

            # torch.distributed will expect a few environment variable to be here. We set the ones common to each
            # process here (the other ones will be set be the launcher).
            with patch_environment(**patched_env):
                # First dummy launch
                if os.environ.get("ACCELERATE_DEBUG_MODE", "false").lower() == "true":
                    launcher = PrepareForLaunch(test_launch, distributed_type="MULTI_GPU")
                    try:
                        start_processes(launcher, args=(), nprocs=num_processes, start_method="fork")
                    except ProcessRaisedException as e:
                        err = "An issue was found when verifying a stable environment for the notebook launcher."
                        if "Cannot re-initialize CUDA in forked subprocess" in e.args[0]:
                            raise RuntimeError(
                                f"{err}"
                                "This likely stems from an outside import causing issues once the `notebook_launcher()` is called. "
                                "Please review your imports and test them when running the `notebook_launcher()` to identify "
                                "which one is problematic and causing CUDA to be initialized."
                            ) from e
                        else:
                            raise RuntimeError(f"{err} The following error was raised: {e}") from e
                # Now the actual launch
                launcher = PrepareForLaunch(function, distributed_type="MULTI_GPU")
                print(f"Launching training on {num_processes} GPUs.")
                try:
                    if rdzv_conf is None:
                        rdzv_conf = {}
                    if rdzv_backend == "static":
                        rdzv_conf["rank"] = node_rank
                        if not rdzv_endpoint:
                            rdzv_endpoint = f"{master_addr}:{use_port}"
                    launch_config_kwargs = dict(
                        min_nodes=num_nodes,
                        max_nodes=num_nodes,
                        nproc_per_node=num_processes,
                        run_id=rdzv_id,
                        rdzv_endpoint=rdzv_endpoint,
                        rdzv_backend=rdzv_backend,
                        rdzv_configs=rdzv_conf,
                        max_restarts=max_restarts,
                        monitor_interval=monitor_interval,
                        start_method="fork",
                    )
                    if is_torch_version(">=", ELASTIC_LOG_LINE_PREFIX_TEMPLATE_PYTORCH_VERSION):
                        launch_config_kwargs["log_line_prefix_template"] = log_line_prefix_template
                    elastic_launch(config=LaunchConfig(**launch_config_kwargs), entrypoint=function)(*args)
                except ProcessRaisedException as e:
                    if "Cannot re-initialize CUDA in forked subprocess" in e.args[0]:
                        raise RuntimeError(
                            "CUDA has been initialized before the `notebook_launcher` could create a forked subprocess. "
                            "This likely stems from an outside import causing issues once the `notebook_launcher()` is called. "
                            "Please review your imports and test them when running the `notebook_launcher()` to identify "
                            "which one is problematic and causing CUDA to be initialized."
                        ) from e
                    else:
                        raise RuntimeError(f"An issue was found when launching the training: {e}") from e

        else:
            # No need for a distributed launch otherwise as it's either CPU, GPU or MPS.
            if is_mps_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                print("Launching training on MPS.")
            elif torch.cuda.is_available():
                print("Launching training on one GPU.")
            else:
                print("Launching training on CPU.")
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
            master_addr="127.0.0.1",
            master_port="29500",
            accelerate_mixed_precision="no",
            accelerate_debug_rdv_file=tmp_file.name,
            accelerate_use_cpu="yes",
        ):
            launcher = PrepareForLaunch(function, debug=True)
            start_processes(launcher, args=args, nprocs=num_processes, start_method="fork")
