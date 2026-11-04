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
from types import SimpleNamespace

import torch
from parameterized import parameterized
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from accelerate.test_utils.testing import AccelerateTestCase, require_torch_min_version
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils.fsdp_utils import fsdp2_apply_ac


class DropoutBlock(nn.Sequential):
    def __init__(self, dropout):
        super().__init__(nn.Linear(8, 16), nn.Dropout(dropout), nn.Linear(16, 8))


@require_torch_min_version(version="2.6.0")
class FSDP2ActivationCheckpointingRngTester(AccelerateTestCase):
    @staticmethod
    def _apply_checkpointing(model, offload=False, **kwargs):
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["DropoutBlock"],
            activation_checkpointing=True,
            activation_checkpointing_offload=offload,
        )
        accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
        return fsdp2_apply_ac(accelerator, model, **kwargs)

    @staticmethod
    def _forward_backward(model, inputs):
        torch.manual_seed(42)
        output = model(inputs)
        output.square().sum().backward()
        return output.detach(), [parameter.grad.clone() for parameter in model.parameters()], inputs.grad.clone()

    @parameterized.expand([(0.0, False), (0.5, False), (0.5, True)])
    def test_matches_uncheckpointed_gradients(self, dropout, offload):
        torch.manual_seed(0)
        reference = nn.Sequential(DropoutBlock(dropout), DropoutBlock(dropout))
        checkpointed = self._apply_checkpointing(copy.deepcopy(reference), offload=offload)
        inputs = torch.randn(4, 8)

        expected_output, expected_grads, expected_input_grad = self._forward_backward(
            reference, inputs.clone().requires_grad_(True)
        )
        output, grads, input_grad = self._forward_backward(checkpointed, inputs.clone().requires_grad_(True))

        torch.testing.assert_close(output, expected_output)
        for actual, expected in zip(grads, expected_grads):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(input_grad, expected_input_grad)

    def test_can_disable_rng_preservation(self):
        torch.manual_seed(0)
        reference = nn.Sequential(DropoutBlock(0.5), DropoutBlock(0.5))
        checkpointed = self._apply_checkpointing(copy.deepcopy(reference), preserve_rng_state=False)
        reference = nn.Sequential(*[checkpoint_wrapper(layer, preserve_rng_state=False) for layer in reference])
        inputs = torch.randn(4, 8)

        expected_output, expected_grads, expected_input_grad = self._forward_backward(
            reference, inputs.clone().requires_grad_(True)
        )
        output, grads, input_grad = self._forward_backward(checkpointed, inputs.clone().requires_grad_(True))

        torch.testing.assert_close(output, expected_output)
        for actual, expected in zip(grads, expected_grads):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(input_grad, expected_input_grad)
