# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from safetensors.torch import load_model

from .utils import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    RNG_STATE_NAME,
    SAFE_MODEL_NAME,
    SAFE_WEIGHTS_NAME,
    SAMPLER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    WEIGHTS_NAME,
    get_pretty_name,
    is_cuda_available,
    is_hpu_available,
    is_mlu_available,
    is_musa_available,
    is_neuron_available,
    is_sdaa_available,
    is_torch_version,
    is_torch_xla_available,
    is_xpu_available,
    load,
    save,
)


if is_torch_version(">=", "2.4.0"):
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

from .logging import get_logger
from .state import PartialState


logger = get_logger(__name__)


def _stage_state_dict_for_async_save(obj):
    """
    Recursively detach and copy tensors to CPU memory so a background thread can serialize a checkpoint to disk
    while training continues. Tensors already on the CPU are cloned (they would otherwise alias live parameters,
    which the training loop keeps mutating); tensors on an accelerator are copied to host memory once here, so the
    writer thread never needs to synchronize with the device.
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.detach()
        if obj.device.type == "cpu":
            return obj.clone()
        return obj.to("cpu")
    if isinstance(obj, dict):
        return type(obj)((k, _stage_state_dict_for_async_save(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(_stage_state_dict_for_async_save(v) for v in obj)
    return obj


def capture_accelerator_state(
    output_dir: str,
    model_states: list[dict],
    optimizers: list,
    schedulers: list,
    dataloaders: list,
    process_index: int,
    step: int,
    scaler: Optional[GradScaler] = None,
    safe_serialization: bool = True,
    stage_to_cpu: bool = False,
) -> list[tuple]:
    """
    Snapshots everything [`save_accelerator_state`] persists into in-memory objects, without touching the disk.

    This splits the (potentially slow) staging of a checkpoint from the actual disk writes, so the writes can run in
    a background thread. Pass `stage_to_cpu=True` to recursively copy all tensors to CPU memory first; this is what
    makes an asynchronous save safe, as the writer thread then only reads host memory while training keeps running.

    Args:
        output_dir (`str` or `os.PathLike`):
            The folder the pending writes will target.
        model_states (`List[torch.nn.Module]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        dataloaders (`List[torch.utils.data.DataLoader]`):
            A list of dataloader instances to save their sampler states
        process_index (`int`):
            The current process index in the Accelerator state
        step (`int`):
            The current step in the internal step tracker
        scaler (`torch.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save;
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether the model should be saved using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        stage_to_cpu (`bool`, *optional*, defaults to `False`):
            Whether to recursively copy all tensors to CPU memory before returning.

    Returns:
        `list[tuple]`: pending writes as `(path, obj, safe_serialization, gated)` tuples, to be passed to
        [`write_accelerator_state`]. Entries with `gated=True` are only written by the main process (or by each
        local main process when `save_on_each_node` is set), mirroring `accelerate.utils.save`.
    """
    output_dir = Path(output_dir)
    pending = []

    def add(state, output_file, entry_safe_serialization=False, gated=True):
        if stage_to_cpu:
            state = _stage_state_dict_for_async_save(state)
        pending.append((output_file, state, entry_safe_serialization, gated))

    # Model states
    for i, state in enumerate(model_states):
        weights_name = WEIGHTS_NAME if not safe_serialization else SAFE_WEIGHTS_NAME
        if i > 0:
            weights_name = weights_name.replace(".", f"_{i}.")
        add(state, output_dir.joinpath(weights_name), entry_safe_serialization=safe_serialization)
    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        add(opt.state_dict(), output_dir.joinpath(optimizer_name))
    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        add(scheduler.state_dict(), output_dir.joinpath(scheduler_name))
    # DataLoader states
    from .data_loader import IterableDatasetShard, SeedableRandomSampler

    for i, dataloader in enumerate(dataloaders):
        sampler_name = f"{SAMPLER_NAME}.bin" if i == 0 else f"{SAMPLER_NAME}_{i}.bin"
        output_sampler_file = output_dir.joinpath(sampler_name)
        # Only save if we have our custom sampler
        if isinstance(dataloader.dataset, IterableDatasetShard):
            sampler = dataloader.get_sampler()
            if isinstance(sampler, SeedableRandomSampler):
                add(sampler, output_sampler_file)
        if getattr(dataloader, "use_stateful_dataloader", False):
            dataloader_state_dict_name = "dl_state_dict.bin" if i == 0 else f"dl_state_dict_{i}.bin"
            output_dataloader_state_dict_file = output_dir.joinpath(dataloader_state_dict_name)
            add(dataloader.state_dict(), output_dataloader_state_dict_file, gated=False)

    # GradScaler state
    if scaler is not None:
        add(scaler.state_dict(), output_dir.joinpath(SCALER_NAME), gated=False)
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{process_index}.pkl"
    states["step"] = step
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    if is_xpu_available():
        states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()
    if is_mlu_available():
        states["torch_mlu_manual_seed"] = torch.mlu.get_rng_state_all()
    elif is_sdaa_available():
        states["torch_sdaa_manual_seed"] = torch.sdaa.get_rng_state_all()
    elif is_musa_available():
        states["torch_musa_manual_seed"] = torch.musa.get_rng_state_all()
    if is_hpu_available():
        states["torch_hpu_manual_seed"] = torch.hpu.get_rng_state_all()
    if is_neuron_available():
        states["torch_neuron_manual_seed"] = torch.neuron.get_rng_state_all()
    if is_cuda_available():
        states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    if is_torch_xla_available():
        states["xm_seed"] = xm.get_rng_state()
    add(states, output_dir.joinpath(states_name), gated=False)
    return pending


def write_accelerator_state(pending: list[tuple], save_on_each_node: bool = False) -> None:
    """
    Writes the pending entries produced by [`capture_accelerator_state`] to disk.

    Args:
        pending (`List[tuple]`):
            The `(path, obj, safe_serialization, gated)` tuples returned by [`capture_accelerator_state`].
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            Whether gated entries should be written by every node's main process, or only the global main process.
    """
    for output_file, state, entry_safe_serialization, gated in pending:
        if gated:
            save(state, output_file, save_on_each_node=save_on_each_node, safe_serialization=entry_safe_serialization)
        else:
            torch.save(state, output_file)
        logger.info(f"State saved in {output_file}")


class AsyncCheckpointManager:
    """
    Internal helper that runs checkpoint writes in a single background thread so training can continue while the
    previous checkpoint is still being written to disk.

    A checkpoint save is composed of two parts: *capturing* the state (detaching/copying tensors, cheap enough to do
    inline) and *writing* it to disk (potentially very slow for large models). The manager only ever offloads the
    writing. At most one write is in flight at a time: submitting a new checkpoint first waits for the previous one,
    so a crash mid-training always leaves a complete checkpoint on disk.
    """

    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._future: Optional[Future] = None

    def submit(self, save_fn: Callable[[], None]) -> None:
        """Waits for any in-flight checkpoint write, then runs `save_fn` in the background thread."""
        self.wait()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="accelerate-async-checkpoint")
        self._future = self._executor.submit(save_fn)

    @property
    def is_active(self) -> bool:
        """Whether a checkpoint write is currently in flight."""
        return self._future is not None and not self._future.done()

    def wait(self) -> None:
        """Blocks until the in-flight checkpoint write (if any) completes, re-raising any error it hit."""
        if self._future is None:
            return
        future, self._future = self._future, None
        try:
            future.result()
        except Exception as e:
            raise RuntimeError(f"The asynchronous checkpoint save failed with the error:\n{e}") from e


async_checkpoint_manager = AsyncCheckpointManager()


def save_accelerator_state(
    output_dir: str,
    model_states: list[dict],
    optimizers: list,
    schedulers: list,
    dataloaders: list,
    process_index: int,
    step: int,
    scaler: Optional[GradScaler] = None,
    save_on_each_node: bool = False,
    safe_serialization: bool = True,
) -> Path:
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    <Tip>

    If `safe_serialization` is `True`, models will be saved with `safetensors` while the rest are saved using native
    `pickle`.

    </Tip>

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (`List[torch.nn.Module]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        dataloaders (`List[torch.utils.data.DataLoader]`):
            A list of dataloader instances to save their sampler states
        process_index (`int`):
            The current process index in the Accelerator state
        step (`int`):
            The current step in the internal step tracker
        scaler (`torch.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save;
        save_on_each_node (`bool`, *optional*):
            Whether to save on every node, or only the main node.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    output_dir = Path(output_dir)
    pending = capture_accelerator_state(
        output_dir,
        model_states,
        optimizers,
        schedulers,
        dataloaders,
        process_index,
        step,
        scaler=scaler,
        safe_serialization=safe_serialization,
    )
    write_accelerator_state(pending, save_on_each_node=save_on_each_node)
    return output_dir


def load_accelerator_state(
    input_dir: str,
    models: list,
    optimizers: list,
    schedulers: list,
    dataloaders: list,
    process_index: int,
    scaler: Optional[GradScaler] = None,
    map_location=None,
    load_kwargs: Optional[dict] = None,
    **load_model_func_kwargs,
) -> dict:
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        dataloaders (`List[torch.utils.data.DataLoader]`):
            A list of dataloader instances used in your program
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
        map_location (`str`, *optional*):
            What device to load the optimizer state onto. Should be one of either "cpu" or "on_device".
        load_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the `load` function.
        load_model_func_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the model's `load_state_dict` method.

    Returns:
        `dict`: Contains the `Accelerator` attributes to override while loading the state.
    """
    # stores the `Accelerator` attributes to override
    override_attributes = dict()
    if map_location not in [None, "cpu", "on_device"]:
        raise TypeError(
            "Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`"
        )
    if map_location is None:
        map_location = "cpu"
    elif map_location == "on_device":
        map_location = PartialState().device

    if load_kwargs is None:
        load_kwargs = {}

    input_dir = Path(input_dir)
    # Model states
    for i, model in enumerate(models):
        ending = f"_{i}" if i > 0 else ""
        input_model_file = input_dir.joinpath(f"{SAFE_MODEL_NAME}{ending}.safetensors")
        if input_model_file.exists():
            load_model(model, input_model_file, device=str(map_location), **load_model_func_kwargs)
        else:
            # Load with torch
            input_model_file = input_dir.joinpath(f"{MODEL_NAME}{ending}.bin")
            state_dict = load(input_model_file, map_location=map_location)
            model.load_state_dict(state_dict, **load_model_func_kwargs)
    logger.info("All model weights loaded successfully")

    # Optimizer states
    for i, opt in enumerate(optimizers):
        optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
        input_optimizer_file = input_dir.joinpath(optimizer_name)
        optimizer_state = load(input_optimizer_file, map_location=map_location, **load_kwargs)
        opt.load_state_dict(optimizer_state)
    logger.info("All optimizer states loaded successfully")

    # Scheduler states
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
        input_scheduler_file = input_dir.joinpath(scheduler_name)
        scheduler_state = load(input_scheduler_file, **load_kwargs)
        scheduler.load_state_dict(scheduler_state)
    logger.info("All scheduler states loaded successfully")

    for i, dataloader in enumerate(dataloaders):
        sampler_name = f"{SAMPLER_NAME}.bin" if i == 0 else f"{SAMPLER_NAME}_{i}.bin"
        input_sampler_file = input_dir.joinpath(sampler_name)
        # Only load if we have our custom sampler
        from .data_loader import IterableDatasetShard, SeedableRandomSampler

        if isinstance(dataloader.dataset, IterableDatasetShard):
            sampler = dataloader.get_sampler()
            if isinstance(sampler, SeedableRandomSampler):
                sampler = dataloader.set_sampler(load(input_sampler_file))
        if getattr(dataloader, "use_stateful_dataloader", False):
            dataloader_state_dict_name = "dl_state_dict.bin" if i == 0 else f"dl_state_dict_{i}.bin"
            input_dataloader_state_dict_file = input_dir.joinpath(dataloader_state_dict_name)
            if input_dataloader_state_dict_file.exists():
                state_dict = load(input_dataloader_state_dict_file, **load_kwargs)
                dataloader.load_state_dict(state_dict)
    logger.info("All dataloader sampler states loaded successfully")

    # GradScaler state
    if scaler is not None:
        input_scaler_file = input_dir.joinpath(SCALER_NAME)
        scaler_state = load(input_scaler_file)
        scaler.load_state_dict(scaler_state)
        logger.info("GradScaler state loaded successfully")

    # Random states
    try:
        states = load(input_dir.joinpath(f"{RNG_STATE_NAME}_{process_index}.pkl"))
        if "step" in states:
            override_attributes["step"] = states["step"]
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
        if is_mlu_available():
            torch.mlu.set_rng_state_all(states["torch_mlu_manual_seed"])
        elif is_sdaa_available():
            torch.sdaa.set_rng_state_all(states["torch_sdaa_manual_seed"])
        elif is_musa_available():
            torch.musa.set_rng_state_all(states["torch_musa_manual_seed"])
        if is_hpu_available():
            torch.hpu.set_rng_state_all(states["torch_hpu_manual_seed"])
        if is_neuron_available():
            torch.neuron.set_rng_state_all(states["torch_neuron_manual_seed"])
        if is_cuda_available():
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        if is_torch_xla_available():
            xm.set_rng_state(states["xm_seed"])
        logger.info("All random states loaded successfully")
    except Exception:
        logger.info("Could not load random states")

    return override_attributes


def save_custom_state(obj, path: str, index: int = 0, save_on_each_node: bool = False) -> None:
    """
    Saves the state of `obj` to `{path}/custom_checkpoint_{index}.pkl`
    """
    # Should this be the right way to get a qual_name type value from `obj`?
    save_location = Path(path) / f"custom_checkpoint_{index}.pkl"
    logger.info(f"Saving the state of {get_pretty_name(obj)} to {save_location}")
    save(obj.state_dict(), save_location, save_on_each_node=save_on_each_node)


def load_custom_state(obj, path: str, index: int = 0) -> None:
    """
    Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`. Will always set `weights_only=False` when
    loading the state.
    """
    load_location = f"{path}/custom_checkpoint_{index}.pkl"
    logger.info(f"Loading the state of {get_pretty_name(obj)} from {load_location}")
    obj.load_state_dict(load(load_location, map_location="cpu", weights_only=False))
