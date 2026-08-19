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
import torch
from torch import nn

from accelerate.test_utils import require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils.fsdp_utils import _OffloadedCheckpointWrapper


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


@require_non_cpu
class OffloadedCheckpointWrapperTester(AccelerateTestCase):
    def _grads(self, model, x):
        out = model(x).square().mean()
        out.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    def test_matches_unwrapped_gradients(self):
        torch.manual_seed(0)
        ref = nn.Sequential(*[Block() for _ in range(4)]).cuda()
        wrapped = nn.Sequential(*[_OffloadedCheckpointWrapper(b) for b in ref])
        x = torch.randn(2, 128, 64, device="cuda")

        ref_grads = self._grads(ref, x)
        for p in ref.parameters():
            p.grad = None
        off_grads = self._grads(wrapped, x)

        for name, ref_grad in ref_grads.items():
            wrapped_name = name.replace(".", "._checkpoint_wrapped_module.", 1)
            torch.testing.assert_close(off_grads[wrapped_name], ref_grad)

    def test_input_storage_freed_and_restored(self):
        torch.manual_seed(0)
        block = _OffloadedCheckpointWrapper(Block().cuda())
        x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
        hidden = x * 1.0  # non-leaf boundary tensor, like a previous layer's output
        out = block(hidden)
        assert hidden.untyped_storage().size() == 0  # offloaded after forward
        out.square().mean().backward()
        assert hidden.untyped_storage().size() != 0  # restored by the recompute
        assert x.grad is not None

    def test_state_dict_hides_the_wrapper(self):
        # A checkpoint written from a wrapped model has to load into an unwrapped one, so the
        # wrapper must not appear in parameter names.
        ref = nn.Sequential(*[Block() for _ in range(2)])
        wrapped = nn.Sequential(*[_OffloadedCheckpointWrapper(Block()) for _ in range(2)])
        # Same keys as torch's own `checkpoint_wrapper`, so a checkpoint written from a wrapped
        # model loads into an unwrapped one.
        assert list(wrapped.state_dict().keys()) == list(ref.state_dict().keys())
        ref.load_state_dict(wrapped.state_dict())

    def test_captured_intermediate_keeps_gradient(self):
        # Reentrant-style checkpointing would detach tensors captured out of the region;
        # this wrapper must preserve their gradient path (grad-enabled forward).
        torch.manual_seed(0)
        block = _OffloadedCheckpointWrapper(CapturingBlock().cuda())
        x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
        captured = []
        out = block(x * 1.0, capture_list=captured)
        loss = out.square().mean() + captured[0].square().mean()
        loss.backward()
        assert x.grad is not None
        assert captured[0].requires_grad
