# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Pipeline parallelism built on top of `torch.distributed.pipelining`.

Unlike tensor parallelism, pipeline parallelism does *not* make a single `nn.Module` behave transparently: every
process only holds a slice of the model (`PipelineStage`) and participates in a microbatch schedule. As a result, the
feature is exposed as a set of standalone utilities rather than through [`accelerate.Accelerator.prepare`]:

```python
>>> from accelerate import Accelerator
>>> from accelerate.pipeline_parallel import prepare_pipeline, pipeline_forward

>>> accelerator = Accelerator()
>>> schedule = prepare_pipeline(model, micro_batch_size=4)
>>> # training loop: runs forward and backward for all microbatches in one call
>>> loss = pipeline_forward(schedule, batch, loss_fn)
>>> accelerator.backward(loss)  # no-op pass-through, gradients were already accumulated
>>> optimizer.step()
```
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch

from .logging import get_logger
from .state import PartialState
from .utils import is_torch_version


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe


__all__ = ["pipeline_forward", "prepare_pipeline"]

logger = get_logger(__name__)


_SUPPORTED_SCHEDULES = ("gpipe", "1f1b")


def _check_pipelining_available():
    if not is_torch_version(">=", "2.4.0"):
        raise RuntimeError("Pipeline parallelism requires torch >= 2.4.0 (module `torch.distributed.pipelining`)")
    try:
        import torch.distributed.pipelining  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Pipeline parallelism requires `torch.distributed.pipelining`, which could not be imported. "
            "Please upgrade your PyTorch installation."
        ) from e


def _default_split_points(model: torch.nn.Module, num_stages: int) -> list[str]:
    """
    Picks sensible default split points by balancing the number of decoder/transformer blocks per stage. Falls back
    to splitting the root module's direct children evenly when the model has no recognizable block containers.
    """
    candidate_paths = ("layers", "blocks", "h", "model.layers", "model.blocks", "transformer.h", "transformer.layers")
    for path in candidate_paths:
        obj = model
        found = True
        for part in path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, (torch.nn.ModuleList, torch.nn.Sequential)) and len(obj) >= num_stages:
            blocks_per_stage = len(obj) / num_stages
            # Stage `i` begins at block `round(i * blocks_per_stage)`; stage 0 starts at the model input itself
            return [f"{path}.{round(i * blocks_per_stage)}" for i in range(1, num_stages)]

    children = list(model.named_children())
    if len(children) >= num_stages:
        children_per_stage = len(children) / num_stages
        return [children[round(i * children_per_stage)][0] for i in range(1, num_stages)]

    raise ValueError(
        f"Could not find a way to split the model into {num_stages} stages automatically; please pass `split_points` "
        'explicitly, e.g. `split_points={"layers.4", "layers.8"}`.'
    )


def _example_inputs(model: torch.nn.Module, micro_batch_size: int, args_for_split, kwargs_for_split):
    """
    Builds the dummy tracing inputs for `torch.distributed.pipelining.pipeline`. Only shapes/dtypes matter here; the
    traced graph is symbolic. Defaults to integer token ids of shape `(micro_batch_size, 128)`, which works for the
    vast majority of decoder-only / encoder-decoder language models.
    """
    if args_for_split is not None or kwargs_for_split:
        return args_for_split or (), kwargs_for_split or {}
    vocab = 32000
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        vocab = getattr(embedding, "num_embeddings", vocab)
    seq_len = 128
    dummy = torch.randint(0, max(vocab, 2), (micro_batch_size, seq_len))
    return (dummy,), {}


def _forward_param_names(model: torch.nn.Module) -> list[str]:
    """Names of the input parameters of the model's forward, self excluded."""
    try:
        params = inspect.signature(model.forward).parameters
        return [name for i, (name, p) in enumerate(params.items()) if i > 0 or p.name != "self"]
    except (ValueError, TypeError):  # pragma: no cover
        return []


def _get_stage(schedule):
    """Returns the `PipelineStage` of a torch pipelining schedule across torch versions."""
    return getattr(schedule, "_stage", None) or getattr(schedule, "stage", None)


def _set_n_microbatches(schedule, n: int):
    """Sets the number of microbatches on a torch pipelining schedule across torch versions."""
    if hasattr(schedule, "set_n_microbatches"):
        schedule.set_n_microbatches(n)
    else:
        schedule._n_microbatches = n


def prepare_pipeline(
    model: torch.nn.Module,
    num_stages: Optional[int] = None,
    micro_batch_size: int = 1,
    pp_rank: Optional[int] = None,
    schedule: Union[str, ScheduleGPipe, Schedule1F1B] = "1f1b",
    split_spec: Optional[dict] = None,
    split_points: Optional[list[str]] = None,
    loss_fn: Optional[Callable] = None,
    group: Optional[ProcessGroup] = None,
    args_for_split: Optional[tuple] = None,
    kwargs_for_split: Optional[dict] = None,
) -> Union[ScheduleGPipe, Schedule1F1B]:
    """
    Splits `model` into pipeline stages and builds the microbatch schedule for the current process.

    Tip: to avoid holding the *full* model in memory on every rank, load it with
    `accelerate.big_modeling.init_empty_weights` (meta device) before calling this function. `pipeline` only needs
    the module structure and weights of the local stage, which are materialized when the stage is built.

    Args:
        model (`torch.nn.Module`):
            The full (unsharded) model. Every rank must receive an identical copy; only the slice belonging to this
            rank participates in the pipeline.
        num_stages (`int`, *optional*):
            The number of pipeline stages, i.e. how many pieces the model is cut into. Defaults to the number of
            processes in `group` (one stage per rank).
        micro_batch_size (`int`, defaults to `1`):
            The per-microbatch batch size. The global batch passed to [`pipeline_forward`] is chunked into
            microbatches of this size; the number of microbatches is then inferred automatically at runtime.
        pp_rank (`int`, *optional*):
            This rank's position in the pipeline. Defaults to the rank within `group`.
        schedule (`str` or torch schedule, defaults to `"1f1b"`):
            Either `"gpipe"`, `"1f1b"`, or a pre-built instance of `torch.distributed.pipelining.ScheduleGPipe` /
            `Schedule1F1B`.
        split_spec (`dict[str, str]`, *optional*):
            A `torch.distributed.pipelining` split spec, e.g. `{"layers.4": "beginning"}`. Takes precedence over
            `split_points`.
        split_points (`list[str]`, *optional*):
            Module qualified names at whose start the pipeline is cut, e.g. `["layers.4", "layers.8"]`. If neither
            this nor `split_spec` is passed, the model's decoder blocks are split evenly across the stages.
        loss_fn (`Callable`, *optional*):
            The loss function, called as `loss_fn(outputs, targets)`. When provided, [`pipeline_forward`]
            accumulates gradients into this rank's stage parameters. `targets` is taken from the `"labels"` or
            `"targets"` key of the batch (or `target=` argument).
        group (`torch.distributed.ProcessGroup`, *optional*):
            The process group spanning the pipeline. Defaults to the default (world) group; pass an explicit group
            to combine pipeline parallelism with data parallelism, e.g. one pipeline group per DP replica.
        args_for_split (`tuple`, *optional*):
            Example positional inputs used to trace the model (only shapes/dtypes matter). If not passed, dummy token
            ids of shape `(micro_batch_size, 128)` are used.
        kwargs_for_split (`dict`, *optional*):
            Example keyword inputs used to trace the model.

    Returns:
        A `torch.distributed.pipelining` schedule object (e.g. `Schedule1F1B`). Pass it to [`pipeline_forward`]
        in your training loop.

    Example:

    ```python
    >>> from accelerate import Accelerator
    >>> from accelerate.pipeline_parallel import prepare_pipeline, pipeline_forward

    >>> accelerator = Accelerator()
    >>> schedule = prepare_pipeline(model, micro_batch_size=8)
    >>> optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters()))
    >>> for batch in dataloader:
    ...     loss = pipeline_forward(schedule, batch, loss_fn)
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    ```
    """
    _check_pipelining_available()
    if isinstance(schedule, str) or split_spec is not None or split_points is not None:
        from torch.distributed.pipelining import SplitPoint, pipeline

    state = PartialState()
    if group is None:
        group_size, group_rank = state.num_processes, state.process_index
    else:
        group_size, group_rank = torch.distributed.get_world_size(group), torch.distributed.get_rank(group)
    num_stages = num_stages if num_stages is not None else group_size

    if num_stages < 2:
        raise ValueError(f"Pipeline parallelism needs at least 2 stages, got num_stages={num_stages}")
    if group_size < num_stages:
        raise ValueError(f"Pipeline group size ({group_size}) cannot be smaller than num_stages ({num_stages})")
    if pp_rank is None:
        pp_rank = group_rank
    if pp_rank >= num_stages:
        raise ValueError(f"pp_rank ({pp_rank}) must be smaller than num_stages ({num_stages})")

    if not isinstance(schedule, str):
        # User brought their own schedule; just attach our metadata and return it
        schedule._accelerate_micro_batch_size = micro_batch_size
        return schedule

    schedule_name = schedule.lower()
    if schedule_name not in _SUPPORTED_SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}, must be one of {_SUPPORTED_SCHEDULES}")

    mb_args, mb_kwargs = _example_inputs(model, micro_batch_size, args_for_split, kwargs_for_split)
    if split_spec is None:
        split_spec = {name: SplitPoint.BEGINNING for name in (split_points or _default_split_points(model, num_stages))}
    pipe = pipeline(model, mb_args=mb_args, mb_kwargs=mb_kwargs, split_spec=split_spec)
    stage = pipe.build_stage(pp_rank, device=state.device, group=group)

    # `n_microbatches` depends on the runtime global batch size, so we seed it with the minimum valid value
    # (`num_stages` for 1F1B) and infer the real value on each `pipeline_forward` call.
    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe

    if schedule_name == "gpipe":
        pipeline_schedule = ScheduleGPipe(stage, n_microbatches=1, loss_fn=loss_fn)
    else:
        pipeline_schedule = Schedule1F1B(stage, n_microbatches=num_stages, loss_fn=loss_fn)

    # torch requires the runtime `step()` inputs to match the traced signature: later stages receive activations
    # positionally, so dict batches are mapped onto the model's forward parameter order (see `pipeline_forward`).
    pipeline_schedule._accelerate_arg_names = _forward_param_names(model)[: len(mb_args)]
    pipeline_schedule._accelerate_kwargs_form = bool(mb_kwargs)
    pipeline_schedule._accelerate_micro_batch_size = micro_batch_size
    # Activations are exchanged with fixed-size p2p sends/recv derived from the traced shapes, so runtime inputs
    # with different shapes deadlock the schedule instead of erroring. We validate them in `pipeline_forward`.
    pipeline_schedule._accelerate_traced_shapes = (
        [tuple(t.shape) for t in mb_args if isinstance(t, torch.Tensor)],
        {k: tuple(t.shape) for k, t in mb_kwargs.items() if isinstance(t, torch.Tensor)},
    )
    schedule = pipeline_schedule
    logger.info(
        f"Pipeline stage {pp_rank}/{num_stages} built on {state.device} with {len(split_spec)} split points "
        f"(schedule={schedule_name}, micro_batch_size={micro_batch_size})"
    )
    return schedule


def _validate_runtime_shapes(schedule, args: tuple, kwargs: dict, n_microbatches: int) -> None:
    """
    Checks that the runtime inputs have the same shapes as the dummy inputs the model was traced with. A mismatch
    would otherwise deadlock the pipeline (stage A sends activations sized by the traced shape while stage B receives
    a different number of elements), so we fail early with an actionable error instead.
    """
    traced_args, traced_kwargs = getattr(schedule, "_accelerate_traced_shapes", ([], {}))
    if not traced_args and not traced_kwargs:
        return
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and i < len(traced_args):
            expected = (n_microbatches * traced_args[i][0],) + tuple(traced_args[i][1:])
            if tuple(arg.shape) != tuple(expected):
                raise ValueError(
                    f"Runtime input #{i} has shape {tuple(arg.shape)} but the pipeline was traced with micro-batch "
                    f"shape {tuple(traced_args[i])} ({n_microbatches} microbatches -> expected global shape "
                    f"{tuple(expected)}). Pass matching `args_for_split`/`kwargs_for_split` to `prepare_pipeline`, "
                    "or adjust `micro_batch_size`/the batch shapes."
                )
    for key, kwarg in kwargs.items():
        if isinstance(kwarg, torch.Tensor) and key in traced_kwargs:
            expected = (n_microbatches * traced_kwargs[key][0],) + tuple(traced_kwargs[key][1:])
            if tuple(kwarg.shape) != tuple(expected):
                raise ValueError(
                    f"Runtime input {key!r} has shape {tuple(kwarg.shape)} but the pipeline was traced with "
                    f"micro-batch shape {tuple(traced_kwargs[key])} ({n_microbatches} microbatches -> expected "
                    f"global shape {tuple(expected)}). Pass matching `args_for_split`/`kwargs_for_split` to "
                    "`prepare_pipeline`, or adjust `micro_batch_size`/the batch shapes."
                )


def _infer_n_microbatches(schedule, args: tuple, kwargs: dict, num_microbatches: Optional[int]) -> int:
    """Determines the number of microbatches for this iteration if not given explicitly."""
    if num_microbatches is not None:
        return num_microbatches
    micro_batch_size = getattr(schedule, "_accelerate_micro_batch_size", None)
    if micro_batch_size is None:
        return schedule._n_microbatches
    batch = args[0] if args else (next(iter(kwargs.values())) if kwargs else None)
    global_batch_size = getattr(batch, "shape", [0])[0] if batch is not None else 0
    if not global_batch_size:
        return schedule._n_microbatches
    return max(1, math.ceil(global_batch_size / micro_batch_size))


def pipeline_forward(
    schedule: Union[ScheduleGPipe, Schedule1F1B],
    batch: Union[dict, tuple, list, torch.Tensor],
    loss_fn: Optional[Callable] = None,
    num_microbatches: Optional[int] = None,
    target=None,
) -> Union[torch.Tensor, tuple, dict, None]:
    """
    Runs one full pipeline iteration (all microbatches) for the current rank.

    With a `loss_fn` (either passed here or at [`prepare_pipeline`] time), gradients are accumulated into the
    stage's parameters on every rank, so `optimizer.step()` can be called right after. Without a `loss_fn`, returns
    the outputs of the last pipeline stage (`None` on all other ranks).

    Args:
        schedule:
            The schedule object returned by [`prepare_pipeline`].
        batch (`dict`, `tuple`, `list` or `torch.Tensor`):
            The global batch for this iteration. Dict batches are passed as model keyword arguments (e.g.
            HuggingFace `transformers` models), other batches as positional arguments. It is chunked into
            microbatches automatically.
        loss_fn (`Callable`, *optional*):
            Overrides the loss function given to [`prepare_pipeline`]. Receives the model outputs; targets are taken
            from the batch's `"labels"`/`"targets"` key or the `target` argument.
        num_microbatches (`int`, *optional*):
            The number of microbatches to split the global batch into. By default, inferred from the batch size and
            the `micro_batch_size` given to [`prepare_pipeline`].
        target (*optional*):
            The targets for the loss function, as an alternative to a `"labels"`/`"targets"` key in `batch`.

    Returns:
        The mean loss (`torch.Tensor`) when a loss function is used (only populated on the last pipeline stage,
        `None` elsewhere), otherwise the last-stage output.
    """
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

    # Extract the loss target first, so it never ends up as a model input and the extraction is identical on every
    # rank regardless of the traced input form.
    if target is None and isinstance(batch, dict):
        target = batch.get("labels")
        if target is None:
            target = batch.get("targets")

    args, kwargs = (), {}
    if isinstance(batch, dict):
        if getattr(schedule, "_accelerate_kwargs_form", False):
            # Traced with kwargs: pass the dict through, minus the extracted target
            kwargs = {k: v for k, v in batch.items() if not (k in ("labels", "targets") and v is target)}
        else:
            # Traced positionally: map the dict onto the model's forward parameter order. This is the common
            # HuggingFace-style case, e.g. `pipeline_forward(schedule, {"input_ids": ids, "labels": y})` against a
            # model with `forward(input_ids, ...)`, where the pipeline cut makes every stage after the first take
            # positional activations only.
            arg_names = getattr(schedule, "_accelerate_arg_names", None)
            if arg_names:
                missing = [name for name in arg_names if name not in batch]
                if missing:
                    raise ValueError(
                        f"The batch dict is missing the model's forward argument(s) {missing}. Expected keys "
                        f"{arg_names}, got {sorted(batch)}."
                    )
                args = tuple(batch[name] for name in arg_names)
            else:
                args = tuple(batch.values())
    elif isinstance(batch, (tuple, list)):
        args = tuple(batch)
    else:
        args = (batch,)

    n_microbatches = _infer_n_microbatches(schedule, args, kwargs, num_microbatches)
    _set_n_microbatches(schedule, n_microbatches)
    _validate_runtime_shapes(schedule, args, kwargs, n_microbatches)

    # `Schedule.step` chunks inputs into microbatches itself; tell it to split tensors along the batch dimension.
    # Rebuilt on every call so the specs always match the current batch's keys and shapes.
    schedule._args_chunk_spec = tuple(TensorChunkSpec(0) for _ in args)
    schedule._kwargs_chunk_spec = {k: TensorChunkSpec(0) for k in kwargs}

    if loss_fn is not None:
        schedule._loss_fn = loss_fn
        # torch computes `_has_backward` from the loss_fn at construction time; keep it in sync when the loss fn is
        # attached later, otherwise the schedule silently runs forward-only and the ranks deadlock.
        schedule._has_backward = True

    has_loss = schedule._loss_fn is not None
    losses: list[torch.Tensor] = []
    # Only collect losses when there is a loss function; passing an empty `losses` list without one makes torch's
    # last stage raise "Expecting N losses but got 0".
    output = schedule.step(*args, target=target, losses=losses if has_loss else None, **kwargs)
    if has_loss:
        # Only the last pipeline stage computes losses; other ranks get an empty list
        return torch.stack(losses).mean() if losses else None
    return output
