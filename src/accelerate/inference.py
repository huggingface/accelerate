import math
from types import MethodType
from typing import Literal
from functools import update_wrapper
import pickle
import torch

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from .state import PartialState
from .utils import (
    calculate_maximum_sizes,
    convert_bytes,
    infer_auto_device_map,
    send_to_device,
    find_batch_size,
)


ParallelMode = Literal["sequential", "pipeline_parallel"]

def generate_device_map(model, num_processes:int=1):
    """
    Calculates the device map for `model` with an offset for PiPPy
    """
    if num_processes == 1:
        return infer_auto_device_map(
            model, 
            no_split_module_classes=getattr("_no_split_modules", model, [])
        )
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
        max_memory = {i: memory for i in range(num_processes)}
    )
    return device_map

def build_pipeline(model, device_map, args, kwargs) -> PipelineStage:
    """
    Attaches the split points to the model based on `self.device_map`
    and generates a `PipelineStage`. Requires passing in needed
    `args` and `kwargs` as the model needs on the CPU.
    """
    # We need to annotate the split points in the model for PiPPy
    state = PartialState()
    split_points = []
    for i in range(1, state.num_processes):
        split_points.append(next(k for k, v in device_map.items() if v == i))
    annotate_split_points(
        model, 
        {
            split_point: PipeSplitWrapper.SplitPoint.BEGINNING
            for split_point in split_points
        }
    )
    with torch.no_grad():
        pipe = Pipe.from_tracing(
            model,
            num_chunks=state.num_processes,
            example_args=args,
            example_kwargs=kwargs
        )
    stage = PipelineStage(
        pipe,
        state.local_process_index,
        device=state.device
    )

    return stage


def prepare_pippy(model, device_map="auto", example_args=(), example_kwargs={}):
    """
    Wraps `model` for PipelineParallelism
    """
    example_args = send_to_device(example_args, "cpu")
    example_kwargs = send_to_device(example_kwargs, "cpu")
    if device_map == "auto":
        device_map = generate_device_map(model, PartialState().num_processes)
    stage = build_pipeline(model, device_map, example_args, example_kwargs)
    model._original_forward = model.forward
    model._original_call = model.__call__
    model.pippy_stage = stage
    model.forward = model.pippy_stage.forward
    return stage