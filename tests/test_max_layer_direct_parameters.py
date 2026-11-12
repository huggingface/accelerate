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

import pytest
import torch
from torch import nn

from accelerate.utils.modeling import (
    calculate_maximum_sizes,
    compute_module_sizes,
    get_max_layer_size,
    infer_auto_device_map,
)


class ParentWithParameter(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(200, dtype=dtype))
        self.child = nn.Linear(1, 1, dtype=dtype)


class ModelWithDirectParameter(nn.Module):
    def __init__(self, dtype=torch.float32, nested=False):
        super().__init__()
        self.first = nn.Linear(4, 5, dtype=dtype)
        parent = ParentWithParameter(dtype=dtype)
        self.parent = nn.Sequential(parent) if nested else parent


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nested", [False, True])
def test_maximum_size_includes_parent_parameters(device, dtype, nested):
    with torch.device(device):
        model = ModelWithDirectParameter(dtype=dtype, nested=nested)

    item_size = torch.tensor([], dtype=dtype).element_size()
    parameter_name = "parent.0.weight" if nested else "parent.weight"
    assert calculate_maximum_sizes(model) == (227 * item_size, (200 * item_size, [parameter_name]))

    modules = list(model.named_children())
    original = modules.copy()
    assert get_max_layer_size(modules, compute_module_sizes(model), []) == (200 * item_size, [parameter_name])
    assert all(
        name == old_name and module is old_module for (name, module), (old_name, old_module) in zip(modules, original)
    )


@pytest.mark.parametrize("nested", [False, True])
def test_no_split_parent_keeps_its_entire_size(nested):
    model = ModelWithDirectParameter(nested=nested)
    model._no_split_modules = ["ParentWithParameter"]
    parent_name = "parent.0" if nested else "parent"
    assert calculate_maximum_sizes(model) == (908, (808, [parent_name]))


def test_equal_parent_parameters_are_reported_in_order():
    model = ModelWithDirectParameter()
    model.parent.other = nn.Parameter(torch.zeros(200))
    assert calculate_maximum_sizes(model) == (1708, (800, ["parent.weight", "parent.other"]))


def test_larger_leaf_stays_a_whole_layer():
    model = ModelWithDirectParameter()
    model.parent.weight = nn.Parameter(torch.zeros(1))
    model.parent.child = nn.Linear(4, 5)
    # Both linear layers include their bias, rather than being split into parameters.
    assert calculate_maximum_sizes(model) == (204, (100, ["first", "parent.child"]))


def tensor_device(name, device_map):
    while name not in device_map:
        assert "." in name or "" in device_map, f"Missing device for {name}"
        name = name.rpartition(".")[0]
    return device_map[name]


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("fallback_allocation", [False, True])
@pytest.mark.parametrize("budget", [850, 900, 908])
def test_parent_parameter_offload_reserves_enough_cpu_memory(device, fallback_allocation, budget):
    with torch.device(device):
        model = ModelWithDirectParameter()
    device_map = infer_auto_device_map(
        model, max_memory={"cpu": budget}, clean_result=False, fallback_allocation=fallback_allocation
    )
    placement = {name: tensor_device(name, device_map) for name, _ in model.named_parameters()}
    resident_bytes = sum(
        parameter.numel() * parameter.element_size()
        for name, parameter in model.named_parameters()
        if placement[name] == "cpu"
    )
    if budget == 908:
        assert set(placement.values()) == {"cpu"}
        assert resident_bytes == 908
    else:
        assert placement["parent.weight"] == "disk"
        # The 800-byte direct parameter must be able to return from disk beside
        # the permanently resident weights.
        assert resident_bytes + 800 <= budget


@pytest.mark.parametrize("buffer", [False, True])
@pytest.mark.parametrize("budget", [1800, 1850, 1908])
def test_recomputed_reservation_keeps_root_tensors(buffer, budget):
    model = nn.Module()
    model.first = nn.Parameter(torch.zeros(250))
    model.next = nn.Parameter(torch.zeros(25))
    model.child = nn.Linear(1, 1)
    if buffer:
        model.register_buffer("large", torch.zeros(200))
    else:
        model.large = nn.Parameter(torch.zeros(200))

    device_map = infer_auto_device_map(model, max_memory={"cpu": budget}, clean_result=False, offload_buffers=True)
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    placement = {name: tensor_device(name, device_map) for name, _ in tensors}
    resident_bytes = sum(
        tensor.numel() * tensor.element_size() for name, tensor in tensors if placement[name] == "cpu"
    )
    if budget == 1908:
        assert set(placement.values()) == {"cpu"}
        assert resident_bytes == 1908
    else:
        # Once the largest (1000-byte) parameter leaves the pending queue, the
        # next 800-byte tensor must still be included in the reservation.
        assert placement["first"] == "cpu"
        assert placement["large"] == "disk"
        assert resident_bytes + 800 <= budget
