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

import pickle

import pytest
import torch

from accelerate.utils import operations
from accelerate.utils.dataclasses import DistributedType


class _VirtualXlaCollective:
    def __init__(self, values):
        self.values = values
        self.sizes = [len(pickle.dumps(value)) for value in values]
        self.max_size = max(self.sizes, default=0)

    def all_gather(self, tensor, dim=0):
        if tensor.numel() == 1:
            return torch.tensor(self.sizes, dtype=tensor.dtype, device=tensor.device)

        padded = []
        for value, size in zip(self.values, self.sizes):
            payload = torch.frombuffer(pickle.dumps(value), dtype=torch.uint8).clone()
            padded.append(torch.nn.functional.pad(payload, (0, self.max_size - size)))
        return torch.cat(padded).to(tensor.device)


class _VirtualXlaState:
    distributed_type = DistributedType.XLA
    device = torch.device("cpu")
    num_processes = 3


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[], ["rank-1", "extra"], []], ["rank-1", "extra"]),
        (
            [
                {"rank": 0, "metadata": {"tokens": [1, 2]}},
                None,
                {"rank": 2, "metadata": {"text": "a much longer string"}},
            ],
            [
                {"rank": 0, "metadata": {"tokens": [1, 2]}},
                None,
                {"rank": 2, "metadata": {"text": "a much longer string"}},
            ],
        ),
        (["short", "a much longer string", ""], ["short", "a much longer string", ""]),
    ],
)
def test_xla_gather_object_preserves_rank_order(monkeypatch, values, expected):
    collective = _VirtualXlaCollective(values)
    monkeypatch.setattr(operations, "PartialState", lambda: _VirtualXlaState)
    monkeypatch.setattr(operations, "xm", collective, raising=False)

    assert operations.gather_object(values[0]) == expected
