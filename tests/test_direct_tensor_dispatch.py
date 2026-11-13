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

import copy
import warnings

import pytest
import torch
from torch import nn

from accelerate import dispatch_model, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_submodules
from accelerate.utils import check_device_map


class _DirectWeights(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(200, dtype=dtype) / 200)
        self.bias = nn.Parameter(torch.tensor(0.2, dtype=dtype))
        self.register_buffer("scale", torch.tensor(0.1, dtype=dtype))
        self.register_buffer("offset", torch.tensor(0.3, dtype=dtype), persistent=False)
        self.child = nn.Linear(1, 1, dtype=dtype)
        self.fail = False

    def forward(self, value):
        if self.fail:
            raise RuntimeError("intentional forward failure")
        return self.child(value) + self.weight.sum() * self.scale + self.bias + self.offset


class _NestedDirectWeights(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.first = nn.Linear(4, 1, dtype=dtype)
        self.parent = _DirectWeights(dtype)

    def forward(self, value):
        return self.parent(self.first(value))


def _device_map(nested):
    prefix = "parent." if nested else ""
    result = {
        prefix + "weight": "disk",
        prefix + "bias": "cpu",
        prefix + "scale": "disk",
        prefix + "child": "cpu",
    }
    if nested:
        result["first"] = "disk"
    return result


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("offload_buffers", [False, True])
def test_direct_tensor_dispatch_roundtrip(tmp_path, nested, dtype, checkpoint, offload_buffers):
    model_type = _NestedDirectWeights if nested else _DirectWeights
    torch.manual_seed(29)
    original = model_type(dtype)
    inputs = torch.arange(8 if nested else 2, dtype=dtype).reshape(2, -1) / 10

    # Exercise learned weights rather than only factory initialization.
    optimizer = torch.optim.SGD(original.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        original(inputs).square().mean().backward()
        optimizer.step()
    reference = copy.deepcopy(original).eval()
    device_map = _device_map(nested)
    with warnings.catch_warnings(record=True) as caught:
        if checkpoint:
            path = tmp_path / "model.bin"
            torch.save(original.state_dict(), path)
            with init_empty_weights():
                model = model_type(dtype)
            model = load_checkpoint_and_dispatch(
                model,
                str(path),
                device_map=device_map,
                offload_folder=str(tmp_path / "weights"),
                offload_buffers=offload_buffers,
            )
        else:
            model = dispatch_model(
                original,
                device_map=device_map,
                offload_dir=str(tmp_path / "weights"),
                offload_buffers=offload_buffers,
            )
    assert not any("device_map keys" in str(item.message) for item in caught)
    direct = model.parent if nested else model
    assert direct.weight.device.type == "meta"
    assert direct.bias.device.type == "cpu"
    assert direct.scale.device.type == ("meta" if offload_buffers else "cpu")
    assert direct.offset.device.type == "cpu"
    assert all(param.device.type == "cpu" for param in direct.child.parameters())
    if nested:
        assert model.first.weight.device.type == "meta"
    model.eval()
    with torch.no_grad():
        for factor in (1, 2, 0.5):
            torch.testing.assert_close(model(inputs * factor), reference(inputs * factor))
            assert direct.weight.device.type == "meta"
            assert direct.scale.device.type == ("meta" if offload_buffers else "cpu")
            assert all(param.device.type == "cpu" for param in direct.child.parameters())

    if not checkpoint:
        remove_hook_from_submodules(model)
        assert all(param.device.type == "cpu" for param in model.parameters())
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), reference(inputs))


class _MeasuredLinear(nn.Linear):
    observer = None

    def forward(self, value):
        if self.observer is not None:
            self.observer()
        return super().forward(value)


class _BudgetParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(200, dtype=torch.float32) / 200)
        self.child = nn.Linear(1, 1)
        self.observer = None

    def forward(self, value):
        if self.observer is not None:
            self.observer()
        return self.child(value) + self.weight.sum()


class _BudgetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = _MeasuredLinear(4, 5)
        self.parent = _BudgetParent()

    def forward(self, value):
        return self.parent(self.first(value).sum(-1, keepdim=True))


def test_split_parent_direct_weight_obeys_live_storage_budget(tmp_path):
    model = _BudgetModel().eval()
    reference = copy.deepcopy(model)
    # This is the split map produced by the planner for the 850-byte CPU budget.
    device_map = {"parent.child": "cpu", "first": "disk", "parent.weight": "disk"}
    model = dispatch_model(model, device_map=device_map, offload_dir=str(tmp_path))
    live_bytes = []

    def observe():
        live_bytes.append(
            sum(param.numel() * param.element_size() for param in model.parameters() if param.device.type == "cpu")
        )

    model.first.observer = observe
    model.parent.observer = observe
    with torch.no_grad():
        for inputs in (torch.ones(2, 4), torch.arange(8, dtype=torch.float32).reshape(2, 4)):
            torch.testing.assert_close(model(inputs), reference(inputs))
    assert live_bytes == [108, 808, 108, 808]
    assert max(live_bytes) <= 850
    assert (tmp_path / "index.json").is_file()


def test_direct_tensor_hook_removal_after_forward_failure(tmp_path):
    model = _DirectWeights()
    reference = copy.deepcopy(model)
    model = dispatch_model(model, _device_map(False), offload_dir=str(tmp_path))
    model.fail = True
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        model(torch.ones(2, 1))
    # As with module offload hooks, removing hooks after a failed call must restore usable weights.
    remove_hook_from_submodules(model)
    model.fail = False
    assert all(param.device.type == "cpu" for param in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(model(torch.ones(2, 1)), reference(torch.ones(2, 1)))


class _TiedDirectWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 3)
        self.right = nn.Linear(3, 3)
        self.right.weight = self.left.weight

    def forward(self, value):
        return self.right(self.left(value))


def test_direct_tensor_dispatch_tied_weights(tmp_path):
    model = _TiedDirectWeights().eval()
    reference = copy.deepcopy(model)
    device_map = {"left.weight": "disk", "left.bias": "cpu", "right.weight": "disk", "right.bias": "cpu"}
    model = dispatch_model(model, device_map, offload_dir=str(tmp_path))
    assert model.left.weight.device.type == model.right.weight.device.type == "meta"
    with torch.no_grad():
        for value in (torch.ones(2, 3), torch.arange(6, dtype=torch.float32).reshape(2, 3)):
            torch.testing.assert_close(model(value), reference(value))
            assert model.left.weight.device.type == model.right.weight.device.type == "meta"


@pytest.mark.parametrize("offload_buffers", [False, True])
def test_whole_module_dispatch_remains_unchanged(tmp_path, offload_buffers):
    model = _NestedDirectWeights().eval()
    reference = copy.deepcopy(model)
    model = dispatch_model(
        model,
        {"parent": "disk", "first": "cpu"},
        offload_dir=str(tmp_path),
        offload_buffers=offload_buffers,
    )
    with torch.no_grad():
        for _ in range(2):
            torch.testing.assert_close(model(torch.ones(2, 4)), reference(torch.ones(2, 4)))
    assert model.parent.weight.device.type == "meta"
    assert model.parent.child.weight.device.type == "meta"
    remove_hook_from_submodules(model)
    assert all(param.device.type == "cpu" for param in model.parameters())


def test_device_map_accepts_registered_tensors_and_warns_on_unknown_names():
    model = _DirectWeights()
    with warnings.catch_warnings(record=True) as caught:
        check_device_map(model, _device_map(False))
    assert not caught
    with pytest.warns(UserWarning, match="unknown"):
        check_device_map(model, {**_device_map(False), "unknown": "cpu"})
