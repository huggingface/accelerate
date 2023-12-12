import math
from functools import partial
from typing import Literal

import torch
from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage
from torch import nn

from .big_modeling import dispatch_model
from .state import PartialState
from .utils import (
    calculate_maximum_sizes,
    convert_bytes,
    infer_auto_device_map,
)


ParallelMode = Literal["sequential", "pipeline_parallel"]


class InferenceHandler:
    """
    Base class for handling different backends for `device_map="auto"` inference.

    Supports the native accelerate version as well as utilizing PiPPy.
    """

    def __init__(self, device_map: str = "auto", parallel_mode: ParallelMode = "sequential"):
        self.model = None
        self.device_map = device_map
        self.state = PartialState()
        self.parallel_mode = parallel_mode
        # if parallel_mode == "pipeline_parallel" and not is_pippy_available():
        #     raise RuntimeError("PiPPy is not installed, but is required for pipeline parallel inference.")

        # Attrs for native pipeline parallelism
        self.pipeline = None
        self.scheduler = None

    @staticmethod
    def generate_device_map(model: nn.Module, parallel_mode: str, num_processes: int = 1):
        if parallel_mode == "sequential":
            # No change or adjustment is needed
            return infer_auto_device_map(
                model,
                no_split_module_classes=model._no_split_modules,
                clean_result=False,
            )
        elif parallel_mode == "pipeline_parallel":
            # Calculate the maximum size of the model weights
            model_size, shared = calculate_maximum_sizes(model)
            # Split memory based on the number of devices
            memory = (model_size + shared[0]) / num_processes
            memory = convert_bytes(memory)
            value, ending = memory.split(" ")

            # Add a chunk to deal with placement issues on `pippy`
            memory = math.ceil(float(value)) * 1.1
            memory = f"{memory} {ending}"
            # Create a device map based on the memory
            max_memory = {i: memory for i in range(num_processes)}
            return infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=model._no_split_modules,
                clean_result=False,
            )
        else:
            raise ValueError(
                f"Unknown parallel mode: {parallel_mode}. Expected either `sequential` or `pipeline_parallel`."
            )

    def prepare_pippy(self, model: nn.Module):
        """
        Prepares a model for inference based on `device_map` for pipeline parallelism.
        """
        if self.device_map == "auto":
            self.device_map = self.generate_device_map(model, "pipeline_parallel", self.state.num_processes)
        model._original_forward = model.forward
        # get all the split points for each device
        split_points = []
        for i in range(1, self.state.num_processes):
            split_points.append(next(k for k, v in self.device_map.items() if v == i))

        # Annotate the model with the split points
        for split_point in split_points:
            annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING})

        model._original_call = model.__call__
        model._original_forward = model.forward
        model.__call__ = partial(self.__call__, model=model)
        return model

    def prepare_native(self, model: nn.Module):
        """
        Prepares a model for inference based on `device_map` for sequential parallelism.
        """
        if self.device_map == "auto":
            self.device_map = self.generate_device_map(model, "sequential", self.state.num_processes)

        model = dispatch_model(model, self.device_map)
        return model

    def prepare(self, model: nn.Module):
        """
        Prepares a model for inference based on `device_map` and `parallel_mode`.
        """
        if self.parallel_mode == "sequential":
            model = self.prepare_native(model)
        elif self.parallel_mode == "pipeline_parallel":
            # Send the model to the device for now
            model.to(self.device)
            model = self.prepare_pippy(model)
        return model

    @property
    def device(self):
        return self.state.device

    def __call__(self, model, *args, **kwargs):
        if model is None:
            raise RuntimeError("Model must be prepared before inference is performed. Please call `prepare` first.")

        with torch.inference_mode():
            if self.parallel_mode == "sequential":
                return model(*args, **kwargs)
            elif self.parallel_mode == "pipeline_parallel":
                if self.pipeline is None:
                    # We need to do our first trace quickly over the model
                    self.pipeline = Pipe.from_tracing(
                        model,
                        num_chunks=self.state.num_processes,
                        example_args=args,
                        example_kwargs=kwargs,
                    )
                if self.scheduler is None:
                    # Create a schedule runtime
                    self.scheduler = PipelineStage(
                        self.pipeline,
                        self.state.local_process_index,
                        device=self.state.device,
                    )
                if self.state.is_local_main_process:
                    # convert kwargs to a tuple, this has been fixed on main
                    args = tuple(kwargs.values())
                    return self.scheduler(*args)
                else:
                    return self.scheduler(*())
