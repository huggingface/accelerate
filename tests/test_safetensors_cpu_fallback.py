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

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from accelerate.utils.modeling import load_state_dict


DEVICE_MAPS = [
    {"a": "cpu", "b": "disk"},
    {"b": "disk", "a": "cpu"},
    {"a": "disk", "b": "cpu"},
    {"absent": "disk", "a": "cpu"},
    {"a": "cpu", "a.child": "cpu", "b": "disk"},
    {"a.child": "disk", "a": "cpu"},
    {"": "cpu", "b": "disk"},
    {},
    {"": "cpu"},
    {"": "disk"},
    None,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device_map", DEVICE_MAPS)
def test_safetensors_fallback_preserves_all_weights(tmp_path, dtype, device_map):
    state = {
        "a.weight": torch.arange(12, dtype=dtype).reshape(3, 4),
        "a.child.weight": torch.arange(6, dtype=dtype).reshape(2, 3),
        "a_other.weight": torch.arange(3, dtype=dtype),
        "b.bias": torch.tensor([0.125, -0.25], dtype=dtype),
        "unassigned": torch.tensor(3.5, dtype=dtype),
        "flags": torch.tensor([True, False]),
        "empty": torch.empty(0, 2, dtype=dtype),
    }
    path = str(tmp_path / "model.safetensors")
    save_file(state, path, metadata={"format": "pt"})
    original_map = copy.deepcopy(device_map)
    actual = load_state_dict(path, device_map=device_map)
    expected = load_file(path)
    assert actual.keys() == expected.keys()
    assert device_map == original_map
    for name, value in expected.items():
        assert actual[name].device.type == "cpu"
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)


class _CheckpointModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.a = torch.nn.Linear(3, 5, dtype=dtype)
        self.b = torch.nn.Linear(5, 2, dtype=dtype)
        self.register_buffer("scale", torch.tensor(0.75, dtype=dtype))

    def forward(self, values):
        return self.b(self.a(values).tanh()) * self.scale


def _assert_optimizer_state_equal(actual, expected):
    actual = actual.state_dict()
    expected = expected.state_dict()
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for index, expected_state in expected["state"].items():
        assert actual["state"][index].keys() == expected_state.keys()
        for name, value in expected_state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual["state"][index][name], value, rtol=0, atol=0)
            else:
                assert actual["state"][index][name] == value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
def test_safetensors_fallback_preserves_resumed_training(tmp_path, dtype, optimizer_type):
    torch.manual_seed(39)
    uninterrupted = _CheckpointModel(dtype)
    optimizer_kwargs = {"lr": 0.03}
    if optimizer_type is torch.optim.SGD:
        optimizer_kwargs["momentum"] = 0.9
    optimizer = optimizer_type(uninterrupted.parameters(), **optimizer_kwargs)
    batches = [(torch.randn(4, 3, dtype=dtype), torch.randn(4, 2, dtype=dtype)) for _ in range(5)]
    for inputs, targets in batches[:2]:
        optimizer.zero_grad()
        (uninterrupted(inputs) - targets).square().mean().backward()
        optimizer.step()

    model_path = str(tmp_path / "model.safetensors")
    optimizer_path = tmp_path / "optimizer.pt"
    save_file(uninterrupted.state_dict(), model_path, metadata={"format": "pt"})
    torch.save(optimizer.state_dict(), optimizer_path)
    resumed = _CheckpointModel(dtype)
    resumed.load_state_dict(load_state_dict(model_path, {"a": "cpu", "b": "disk"}), strict=True)
    resumed_optimizer = optimizer_type(resumed.parameters(), **optimizer_kwargs)
    resumed_optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

    for inputs, targets in batches[2:]:
        losses = []
        for model, opt in [(uninterrupted, optimizer), (resumed, resumed_optimizer)]:
            opt.zero_grad()
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for original, restored in zip(uninterrupted.parameters(), resumed.parameters()):
            torch.testing.assert_close(original.grad, restored.grad, rtol=0, atol=0)
        optimizer.step()
        resumed_optimizer.step()
        for name, value in uninterrupted.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
        _assert_optimizer_state_equal(resumed_optimizer, optimizer)
