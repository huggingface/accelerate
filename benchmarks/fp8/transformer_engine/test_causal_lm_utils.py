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

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from causal_lm_utils import PackedDataset, evaluate_model, pack_tokens


def test_packing_preserves_order_and_discards_only_incomplete_sequence():
    packed = pack_tokens(list(range(35)), 16)
    assert packed.shape == (2, 16)
    assert torch.equal(packed.flatten(), torch.arange(32))
    dataset = PackedDataset(packed)
    assert len(dataset) == 2
    assert torch.equal(dataset[1]["input_ids"], torch.arange(16, 32))
    assert torch.equal(dataset[1]["labels"], dataset[1]["input_ids"])


def test_packing_requires_one_complete_sequence():
    with pytest.raises(ValueError, match="complete sequence"):
        pack_tokens([1, 2, 3], 16)


class Model(torch.nn.Module):
    def forward(self, input_ids, labels):
        assert not self.training
        assert not torch.is_grad_enabled()
        logits = torch.zeros((*input_ids.shape, 3))
        logits[..., 1] = 1
        loss = torch.tensor(1.0 if input_ids[0, 0] == 1 else 3.0)
        return SimpleNamespace(loss=loss, logits=logits)


def test_evaluation_weights_loss_by_shifted_token_count_and_restores_mode():
    model = Model().train()
    batches = []
    for token, length in ((1, 17), (2, 33)):
        ids = torch.full((1, length), token)
        batches.append({"input_ids": ids, "labels": ids})
    with patch("causal_lm_utils.torch.autocast", return_value=nullcontext()):
        result = evaluate_model(model, batches)
    assert result["loss"] == pytest.approx(7 / 3)
    assert result["perplexity"] == pytest.approx(math.exp(7 / 3))
    assert result["token_accuracy"] == pytest.approx(1 / 3)
    assert model.training
