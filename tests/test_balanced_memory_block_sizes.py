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

from collections import defaultdict

import pytest
import torch
from torch import nn

from accelerate.utils.modeling import check_device_map, get_balanced_memory, infer_auto_device_map


class ResidualBlock(nn.Module):
    def __init__(self, hidden, device):
        super().__init__()
        self.in_proj = nn.Linear(8, hidden, bias=False, device=device)
        self.out_proj = nn.Linear(hidden, 8, bias=False, device=device)

    def forward(self, value):
        return value + self.out_proj(torch.relu(self.in_proj(value)))


class OtherResidualBlock(ResidualBlock):
    pass


def make_model(widths, device, alternating=False):
    model = nn.Module()
    model.blocks = nn.ModuleList(
        (OtherResidualBlock if alternating and index % 2 else ResidualBlock)(width, device)
        for index, width in enumerate(widths)
    )
    return model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("widths", [(1, 16, 16, 1), (16, 1, 1, 16), (1, 1, 16, 16)])
@pytest.mark.parametrize("container", [list, set])
def test_buffer_uses_largest_instance_independently_of_module_order(device, widths, container):
    model = make_model(widths, device)
    budgets = {0: 10000, 1: 10000, "cpu": 10000}
    balanced = get_balanced_memory(
        model, max_memory=budgets, no_split_module_classes=container(["ResidualBlock"])
    )
    # 2,176 parameter bytes, a 1,024-byte largest block and a 272-byte mean leaf.
    # The existing 1.25 buffer multiplier gives 1,088 + 1,280 = 2,368 bytes.
    assert balanced == {0: 2368, 1: 10000, "cpu": 10000}
    assert budgets == {0: 10000, 1: 10000, "cpu": 10000}
    assert all(parameter.device.type == device for parameter in model.parameters())


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    ("dtype", "special_dtypes", "expected"),
    [
        (torch.float16, None, 1184),
        (None, {"blocks.1.in_proj.weight": torch.float64}, 3264),
    ],
)
def test_largest_block_respects_planned_weight_dtypes(device, dtype, special_dtypes, expected):
    model = make_model((1, 16, 16, 1), device)
    balanced = get_balanced_memory(
        model,
        max_memory={0: 10000, 1: 10000},
        no_split_module_classes="ResidualBlock",
        dtype=dtype,
        special_dtypes=special_dtypes,
    )
    # FP16 halves all sizes. The special FP64 weight adds 512 bytes to the
    # largest block: total=2,688 and buffer=1.25*1,536, giving 3,264.
    assert balanced == {0: expected, 1: 10000}


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    ("widths", "classes", "expected"),
    [((4, 4, 4, 4), ["ResidualBlock"], 832), ((1, 16, 16, 1), ["AbsentBlock"], 1428)],
)
def test_equal_blocks_and_absent_classes_keep_existing_budgets(device, widths, classes, expected):
    model = make_model(widths, device)
    balanced = get_balanced_memory(model, max_memory={0: 10000, 1: 10000}, no_split_module_classes=classes)
    assert balanced == {0: expected, 1: 10000}


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("second_device", [1, 2])
def test_balanced_map_avoids_unnecessary_cpu_offload(device, second_device):
    model = make_model((1, 16, 16, 1), device)
    budgets = {0: 2176, second_device: 1500, "cpu": 10000}
    balanced = get_balanced_memory(model, max_memory=budgets, no_split_module_classes=["ResidualBlock"])
    device_map = infer_auto_device_map(
        model, max_memory=balanced, no_split_module_classes=["ResidualBlock"], clean_result=False
    )
    check_device_map(model, device_map)
    assert set(device_map.values()) == {0, second_device}
    assert all(name.count(".") == 1 for name in device_map)
    used = defaultdict(int)
    for name, parameter in model.named_parameters():
        matches = [key for key in device_map if name == key or name.startswith(key + ".")]
        assert len(matches) == 1
        used[device_map[matches[0]]] += parameter.numel() * parameter.element_size()
    assert sum(used.values()) == 2176
    assert all(size <= budgets[target] for target, size in used.items())


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("ids", [(0, 1, 2), (0, 2, 4)])
def test_low_zero_uses_largest_block_on_remaining_devices(device, ids):
    model = make_model((1, 16, 16, 1), device)
    balanced = get_balanced_memory(
        model, max_memory=dict.fromkeys(ids, 10000), no_split_module_classes=["ResidualBlock"], low_zero=True
    )
    assert balanced == {ids[0]: 0, ids[1]: 2368, ids[2]: 10000}


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_scanning_continues_after_every_class_has_been_seen(device):
    model = make_model((1, 2, 16, 16), device, alternating=True)
    balanced = get_balanced_memory(
        model,
        max_memory={0: 10000, 1: 10000},
        no_split_module_classes=["ResidualBlock", "OtherResidualBlock"],
    )
    # 2,240 total bytes and a largest block of 1,024 bytes.
    assert balanced == {0: 2400, 1: 10000}
