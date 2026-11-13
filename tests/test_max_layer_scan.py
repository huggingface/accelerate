# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy

import pytest
import torch
from torch import nn

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import compute_module_sizes, infer_auto_device_map
from accelerate.utils.modeling import get_max_layer_size


class ScanBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.up = nn.Linear(2, 4, bias=False, **kwargs)
        self.down = nn.Linear(4, 2, bias=False, **kwargs)

    def forward(self, inputs):
        return self.down(self.up(inputs).tanh())


class ScanModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.a = nn.Sequential(
            nn.Linear(2, 4, bias=False, **kwargs),
            nn.Tanh(),
            nn.Linear(4, 2, bias=False, **kwargs),
        )
        self.b = ScanBlock(**kwargs)
        self.c = nn.Linear(2, 4, bias=False, **kwargs)

    def forward(self, inputs):
        return self.c(self.b(self.a(inputs)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "no_split,max_elements,names",
    [
        ([], 8, ["a.0", "a.2", "b.up", "b.down", "c"]),
        (["Sequential"], 16, ["a"]),
        (["ScanBlock"], 16, ["b"]),
        (["Sequential", "ScanBlock"], 16, ["a", "b"]),
    ],
)
def test_maximum_layer_scan_preserves_depth_first_order(dtype, device, reverse, no_split, max_elements, names):
    model = ScanModel(dtype=dtype, device=device)
    modules = list(model.named_children())
    if reverse:
        modules.reverse()
        if no_split == []:
            names = ["c", "b.up", "b.down", "a.0", "a.2"]
        elif len(no_split) == 2:
            names = ["b", "a"]
    saved_modules = modules.copy()
    sizes = compute_module_sizes(model)
    saved_sizes = dict(sizes)

    assert get_max_layer_size(modules, sizes, no_split) == (
        max_elements * torch.empty((), dtype=dtype).element_size(),
        names,
    )
    assert modules == saved_modules
    assert sizes == saved_sizes


def test_maximum_layer_scan_keeps_tensor_leaves_and_zero_sizes():
    weight = nn.Parameter(torch.empty(2, 3))
    buffer = torch.empty(6)
    empty = nn.Parameter(torch.empty(0))
    modules = [("weight", weight), ("empty", empty), ("buffer", buffer)]
    assert get_max_layer_size(modules, {"weight": 24, "empty": 0, "buffer": 24}, []) == (
        24,
        ["weight", "buffer"],
    )
    assert get_max_layer_size([("empty", empty)], {"empty": 0}, []) == (0, ["empty"])
    assert get_max_layer_size([], {}, []) == (0, [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("clean_result", [False, True])
@pytest.mark.parametrize("no_split", [[], ["ScanBlock"]])
def test_scan_driven_device_map_preserves_trained_checkpoint_continuation(tmp_path, dtype, clean_result, no_split):
    torch.manual_seed(23)
    reference = ScanModel(dtype=dtype)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.5)
    inputs = torch.tensor([[0.2, -0.4], [0.3, 0.8], [-0.7, 0.1]], dtype=dtype)
    for _ in range(2):
        reference_optimizer.zero_grad()
        reference(inputs).square().mean().backward()
        reference_optimizer.step()
    weights_file = tmp_path / "weights.pt"
    optimizer_file = tmp_path / "optimizer.pt"
    torch.save(reference.state_dict(), weights_file)
    torch.save(reference_optimizer.state_dict(), optimizer_file)
    with init_empty_weights():
        restored = ScanModel(dtype=dtype)
    memory = {"cpu": 1024}
    device_map = infer_auto_device_map(
        restored,
        max_memory=memory,
        no_split_module_classes=no_split,
        clean_result=clean_result,
    )
    assert memory == {"cpu": 1024}
    assert set(device_map.values()) == {"cpu"}
    restored = load_checkpoint_and_dispatch(restored, weights_file, device_map=device_map)
    optimizer = torch.optim.SGD(restored.parameters(), lr=0.03, momentum=0.5)
    optimizer.load_state_dict(torch.load(optimizer_file, weights_only=True))
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    saved_map = deepcopy(device_map)

    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual_output, expected_output = restored(inputs), reference(inputs)
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        actual_output.square().mean().backward()
        expected_output.square().mean().backward()
        for actual, expected in zip(restored.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(restored.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_state, expected_state = (
            optimizer.state_dict(),
            reference_optimizer.state_dict(),
        )
        assert actual_state["param_groups"] == expected_state["param_groups"]
        for index, state in actual_state["state"].items():
            torch.testing.assert_close(
                state["momentum_buffer"],
                expected_state["state"][index]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
        assert device_map == saved_map
