# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    parallelize_module,
)


if TYPE_CHECKING:
    from accelerate import Accelerator


class RouterNoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout=None,
        output_layout=None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (input_layout,), run_check=False)

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(placements=(desired_input_layout,), async_op=True)
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        return (out.to_local() for out in outputs)

    def _apply(self, module: nn.Module, device_mesh) -> nn.Module:
        from functools import partial

        return distribute_module(
            module,
            device_mesh,
            None,
            partial(self._prepare_input_fn, self.input_layout, self.desired_input_layout),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


class PrepareCombinedInputOutput(ParallelStyle):
    def __init__(
        self,
        input_layouts=None,
        desired_input_layouts=None,
        output_layouts=None,
        desired_output_layouts=None,
        use_local_output_in=True,
        use_local_output_out=True,
    ):
        self.prep_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            use_local_output=use_local_output_in,
        )
        self.prep_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output_out,
        )

    def _apply(self, module, device_mesh):
        self.prep_input._apply(module, device_mesh)
        self.prep_output._apply(module, device_mesh)


class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        if name == "":
            module.register_parameter(
                "gate_proj", nn.Parameter(distribute_tensor(module.gate_proj, device_mesh, [Shard(2)]))
            )  # Column-wise sharding
            module.register_parameter(
                "down_proj",
                nn.Parameter(distribute_tensor(module.down_proj, device_mesh, [Shard(1)])),
            )  # Row-wise sharding
            module.register_parameter(
                "up_proj",
                nn.Parameter(distribute_tensor(module.up_proj, device_mesh, [Shard(2)])),
            )  # Column-wise sharding

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (Replicate(),), run_check=False)
        return (input_tensor, *inputs[1:])

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        outputs = outputs.redistribute(placements=(Replicate(),))

        return outputs.to_local()

    def _apply(self, module: nn.Module, device_mesh):
        return distribute_module(
            module, device_mesh, self._partition_fn, self._prepare_input_fn, self._prepare_output_fn
        )


def apply_tp_plan(accelerator: "Accelerator", model: torch.nn.Module):
    tp_plan = getattr(model, "_tp_plan", {})
    base_tp_plan = getattr(model.config, "base_model_tp_plan", {})

    tp_plan = {**base_tp_plan, **tp_plan}

    def prepare_tp_plan() -> dict:
        layer_plan = {}
        base_plan = {}
        for k, v in tp_plan.items():
            if "layer" in k:
                new_plan = layer_plan
                k_modifier = lambda x: x.removeprefix("layers.*.")  # noqa: E731
            else:
                new_plan = base_plan
                k_modifier = lambda x: x  # noqa: E731

            k = k_modifier(k)
            # We're in MOE
            if k.endswith("experts"):
                new_plan[k] = TensorParallel()
                parent_module = k.rsplit(".", 1)[0]
                new_plan[parent_module] = PrepareCombinedInputOutput(
                    input_layouts=Replicate(),
                    desired_input_layouts=Replicate(),
                    use_local_output_in=True,
                    output_layouts=(Replicate(), Replicate()),  # we go out with reduced rowwise - replication
                    desired_output_layouts=(
                        Replicate(),
                        Replicate(),
                    ),  # TODO: we can shard on dim 1 - go in already sharded on seq dim when we add sequence parallel
                )
                new_plan[f"{parent_module}.router"] = (
                    RouterNoParallel()
                )  # Router is not parallelized, we just use it to convert module
            # We're in submodules of Experts - we can skip as they're covered with TensorParallel
            elif "experts" in k:
                continue
            elif "expert" in k and "down_proj" in k:
                new_plan[k] = RowwiseParallel(output_layouts=Replicate())
                continue

            if v == "colwise":
                new_plan[k] = ColwiseParallel()
            elif v == "rowwise":
                new_plan[k] = RowwiseParallel()
            elif v == "colwise_rep":
                new_plan[k] = ColwiseParallel(output_layouts=Replicate())
            elif v == "sequence_parallel":
                pass  # Todo: siro - layers after seq parallel should have inputs set to Shard(1) and desired to (usually) Replicate()
            elif v == "local_colwise":
                new_plan[k] = ColwiseParallel()
            elif v == "local_rowwise":
                new_plan[k] = RowwiseParallel()
            else:
                pass
        return base_plan, layer_plan

    base_tp_plan, layer_tp_plan = prepare_tp_plan()

    parallelize_module(
        model,
        device_mesh=accelerator.torch_device_mesh["tp"],
        parallelize_plan=base_tp_plan,
    )

    for layer in model.model.layers:
        parallelize_module(
            layer,
            device_mesh=accelerator.torch_device_mesh["tp"],
            parallelize_plan=layer_tp_plan,
        )

    return model
