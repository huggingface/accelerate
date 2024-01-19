import math
from types import MethodType
from typing import Literal

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from .state import PartialState
from .utils import (
    calculate_maximum_sizes,
    convert_bytes,
    ignorant_find_batch_size,
    infer_auto_device_map,
    send_to_device,
    slice_and_concatenate,
)


ParallelMode = Literal["sequential", "pipeline_parallel"]


def generate_device_map(model, num_processes: int = 1, no_split_module_classes=None):
    """
    Calculates the device map for `model` with an offset for PiPPy
    """
    if num_processes == 1:
        return infer_auto_device_map(model, no_split_module_classes=no_split_module_classes, clean_result=False)
    model_size, shared = calculate_maximum_sizes(model)

    # Split into `n` chunks for each GPU
    memory = (model_size + shared[0]) / num_processes
    memory = convert_bytes(memory)
    value, ending = memory.split(" ")

    # Add a chunk to deal with potential extra shared memory instances
    memory = math.ceil(float(value)) * 1.1
    memory = f"{memory} {ending}"
    device_map = infer_auto_device_map(
        model,
        max_memory={i: memory for i in range(num_processes)},
        no_split_module_classes=no_split_module_classes,
        clean_result=False,
    )
    return device_map


def build_pipeline(model, split_points, args, kwargs) -> PipelineStage:
    """
    Attaches the split points to the model based on `self.device_map` and generates a `PipelineStage`. Requires passing
    in needed `args` and `kwargs` as the model needs on the CPU.
    """
    # We need to annotate the split points in the model for PiPPy
    state = PartialState()
    annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING for split_point in split_points})
    pipe = Pipe.from_tracing(model, num_chunks=state.num_processes, example_args=args, example_kwargs=kwargs)
    stage = PipelineStage(pipe, state.local_process_index, device=state.device)

    return stage


def pippy_forward(forward, *args, **kwargs):
    state = PartialState()
    output = None

    if state.num_processes == 1:
        output = forward(*args, **kwargs)
    elif state.is_local_main_process:
        found_batch_size = None
        for arg in args:
            found_batch_size = ignorant_find_batch_size(arg)
            if found_batch_size is not None:
                break
        for kwarg in kwargs.values():
            found_batch_size = ignorant_find_batch_size(kwarg)
            if found_batch_size is not None:
                break
        if found_batch_size is None:
            raise ValueError("Could not find batch size from args or kwargs")
        else:
            if (found_batch_size % state.num_processes) != 0:
                slice_to_cut = slice(0, found_batch_size - (found_batch_size % state.num_processes))
                args = slice_and_concatenate(args, slice_to_cut)
                kwargs = slice_and_concatenate(kwargs, slice_to_cut)
        forward(*args, **kwargs)
    elif state.is_last_process:
        output = forward()
    else:
        forward()
    return output


def prepare_pippy(model, split_points="auto", no_split_module_classes=[], example_args=(), example_kwargs={}):
    """
    Wraps `model` for PipelineParallelism
    """
    state = PartialState()
    example_args = send_to_device(example_args, "cpu")
    example_kwargs = send_to_device(example_kwargs, "cpu")
    if split_points == "auto":
        device_map = generate_device_map(model, state.num_processes, no_split_module_classes=no_split_module_classes)
        split_points = []
        for i in range(1, state.num_processes):
            split_points.append(next(k for k, v in device_map.items() if v == i))
    stage = build_pipeline(model, split_points, example_args, example_kwargs)
    model._original_forward = model.forward
    model._original_call = model.__call__
    model.pippy_stage = stage
    model.hf_split_points = split_points

    def forward(*args, **kwargs):
        return pippy_forward(stage.forward, *args, **kwargs)

    # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`
    model_forward = MethodType(forward, model)
    forward.__wrapped__ = model_forward
    model.forward = forward
    return model
