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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    checkpoint_wrapper,
    offload_wrapper,
)

from accelerate.test_utils import require_cuda
from accelerate.test_utils.testing import AccelerateTestCase, require_fsdp2
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils.fsdp_utils import fsdp2_apply_ac


def offloaded(module):
    """What `fsdp2_apply_ac` builds when `activation_checkpointing_offload` is on"""
    return offload_wrapper(checkpoint_wrapper(module))


class Block(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, hidden_states, scale=1.0):
        return hidden_states + self.down(torch.nn.functional.silu(self.up(hidden_states))) * scale


class CapturingBlock(Block):
    """Mimics models that record an intermediate out of the layer (e.g. MoE router logits)."""

    def forward(self, hidden_states, capture_list=None, **kwargs):
        inner = self.up(hidden_states)
        if capture_list is not None:
            capture_list.append(inner)
        return hidden_states + self.down(torch.nn.functional.silu(inner))


class DropoutBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(16, 8)

    def forward(self, hidden_states):
        return hidden_states + self.down(self.dropout(torch.relu(self.up(hidden_states))))


@require_fsdp2
class FSDP2ActivationCheckpointingRNGTest(AccelerateTestCase):
    @parameterized.expand([(False, 0.0), (True, 0.0), (False, 0.1), (True, 0.1)])
    def test_matches_uncheckpointed_training(self, offload, dropout):
        torch.manual_seed(42)
        reference = nn.Sequential(DropoutBlock(dropout), DropoutBlock(dropout))
        checkpointed = copy.deepcopy(reference)
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            activation_checkpointing=True,
            activation_checkpointing_offload=offload,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["DropoutBlock"],
        )
        plugin.set_auto_wrap_policy(checkpointed)
        fsdp2_apply_ac(SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin)), checkpointed)
        self.assertEqual(sum(isinstance(m, CheckpointWrapper) for m in checkpointed.modules()), 2)
        inputs = torch.randn(4, 8)

        def forward_backward(model):
            x = inputs.detach().clone().requires_grad_()
            torch.manual_seed(1234)
            outputs = model(x)
            outputs.square().mean().backward()
            return outputs, [p.grad for p in model.parameters()], x.grad, torch.get_rng_state()

        expected_outputs, expected_grads, expected_input_grad, expected_rng = forward_backward(reference)
        outputs, grads, input_grad, rng = forward_backward(checkpointed)
        torch.testing.assert_close(outputs, expected_outputs)
        self.assertEqual(len(grads), len(expected_grads))
        for grad, expected in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected)
        torch.testing.assert_close(input_grad, expected_input_grad)
        self.assertTrue(torch.equal(rng, expected_rng))


@require_cuda
class ActivationCheckpointingOffloadTester(AccelerateTestCase):
    @staticmethod
    def _grads(model, x):
        """Gradients in parameter order, since wrapping prefixes the parameter names."""
        model(x).square().mean().backward()
        return [p.grad.clone() for p in model.parameters()]

    def test_matches_unwrapped_gradients(self):
        torch.manual_seed(0)
        blocks = [Block().cuda() for _ in range(3)]
        x = torch.randn(2, 16, 64, device="cuda")
        expected = self._grads(nn.Sequential(*blocks), x.clone())
        for block in blocks:
            block.zero_grad(set_to_none=True)
        got = self._grads(nn.Sequential(*[offloaded(block) for block in blocks]), x.clone())
        assert len(got) == len(expected)
        for actual, wanted in zip(got, expected):
            torch.testing.assert_close(actual, wanted)

    def test_saved_activations_move_to_host(self):
        """The point of the option: what the checkpoints saved for their recompute is not on the GPU.

        Measured over a stack, because the tensor saved by the first layer is the caller's input,
        which is allocated before the measurement and so costs nothing extra either way.
        """

        def held_after_forward(wrap):
            torch.manual_seed(0)
            blocks = nn.Sequential(*[wrap(Block().cuda()) for _ in range(4)])
            x = torch.randn(4, 512, 64, device="cuda", requires_grad=True)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            out = blocks(x)  # keep `out` alive, so the graph it holds is counted
            held = torch.cuda.memory_allocated() - before
            out.sum().backward()
            assert x.grad is not None
            return held

        with_offload = held_after_forward(offloaded)
        without = held_after_forward(lambda m: checkpoint_wrapper(m))
        assert with_offload < without, f"offload held {with_offload} bytes, plain held {without}"

    def test_caller_can_still_read_the_input(self):
        """Regression: the wrapper must not free a tensor the caller still holds.

        A model that returns its per-layer inputs (`output_hidden_states=True`) hands out the same
        tensor objects it passes to the layers, and reading them before the backward pass must work.
        """
        torch.manual_seed(0)
        block = offloaded(Block().cuda())
        x = torch.randn(2, 32, 64, device="cuda", requires_grad=True)
        out = block(x)
        assert x.untyped_storage().size() > 0
        assert torch.isfinite(x.float().sum())
        out.sum().backward()

    def test_state_dict_hides_the_wrapper(self):
        torch.manual_seed(0)
        wrapped = nn.Sequential(*[offloaded(Block()) for _ in range(2)])
        reference = nn.Sequential(*[Block() for _ in range(2)])
        assert set(wrapped.state_dict()) == set(reference.state_dict())

    def test_captured_intermediate_keeps_gradient(self):
        torch.manual_seed(0)
        block = offloaded(CapturingBlock().cuda())
        captured = []
        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
        block(x, capture_list=captured).sum().backward()
        assert captured and captured[0].requires_grad
