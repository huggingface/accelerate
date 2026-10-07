# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fp8_utils import evaluate_model


class Model(torch.nn.Module):
    def forward(self, input_ids, labels):
        assert not self.training
        assert not torch.is_grad_enabled()
        return SimpleNamespace(logits=input_ids)


class Metric:
    def __init__(self):
        self.count = 0

    def add_batch(self, predictions, references):
        assert torch.equal(predictions, references)
        self.count += references.numel()

    def compute(self):
        return {"samples": self.count}


@pytest.mark.parametrize("training", [True, False])
def test_evaluation_restores_mode_and_includes_partial_batch(training):
    model = Model().train(training)
    dataset = [{"input_ids": torch.tensor([0.0, 1.0]), "labels": torch.tensor(1)} for _ in range(19)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    with patch("fp8_utils.torch.autocast", return_value=nullcontext()) as autocast:
        result = evaluate_model(model, loader, Metric())
    assert result == {"samples": 19}
    assert model.training == training
    autocast.assert_called_with(device_type="cuda", dtype=torch.bfloat16)


def test_evaluation_restores_training_mode_on_failure():
    model = Model().train()
    dataset = [{"input_ids": torch.tensor([0.0, 1.0]), "labels": torch.tensor(1)}]
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    with patch("fp8_utils.torch.autocast", return_value=nullcontext()):
        with patch.object(model, "forward", side_effect=RuntimeError("evaluation failed")):
            with pytest.raises(RuntimeError, match="evaluation failed"):
                evaluate_model(model, loader, Metric())
    assert model.training
