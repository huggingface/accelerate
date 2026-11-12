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

import pickle

import torch
from parameterized import parameterized

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.test_utils import require_cpu, require_fp16, require_non_cpu
from accelerate.test_utils.testing import AccelerateTestCase


@require_cpu
class CPUOptimizerTester(AccelerateTestCase):
    def test_accelerated_optimizer_pickling(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator()
        optimizer = accelerator.prepare(optimizer)
        try:
            pickle.loads(pickle.dumps(optimizer))
        except Exception as e:
            self.fail(f"Accelerated optimizer pickling failed with {e}")

    @parameterized.expand(
        [
            ("step_pre", "step"),
            ("step_post", "step"),
            ("state_dict_pre", "state_dict"),
            ("state_dict_post", "state_dict"),
            ("load_state_dict_pre", "load_state_dict"),
            ("load_state_dict_post", "load_state_dict"),
        ]
    )
    def test_optimizer_hooks(self, hook_type, operation):
        accelerator = Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.ones(1))
        original_optimizer = torch.optim.SGD([parameter], lr=0.1)
        optimizer = accelerator.prepare(original_optimizer)
        state_dict = optimizer.state_dict()
        calls = []

        def hook(opt, *args):
            self.assertIs(opt, original_optimizer)
            calls.append("prepared")

        register = getattr(optimizer, f"register_{hook_type}_hook")
        existing = getattr(original_optimizer, f"register_{hook_type}_hook")(lambda *args: calls.append("existing"))
        handle = register(hook, prepend=True) if operation != "step" else register(hook)
        parameter.grad = torch.ones_like(parameter)
        args = (state_dict,) if operation == "load_state_dict" else ()
        getattr(optimizer, operation)(*args)
        expected = ["prepared", "existing"] if operation != "step" else ["existing", "prepared"]
        self.assertEqual(calls, expected)

        calls.clear()
        handle.remove()
        getattr(optimizer, operation)(*args)
        self.assertEqual(calls, ["existing"])
        existing.remove()

    def test_optimizer_post_step_hook_with_gradient_accumulation(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer = accelerator.prepare(model, optimizer)
        weights = []
        optimizer.register_step_post_hook(lambda *args: weights.append(model.weight.detach().clone()))

        for _ in range(4):
            with accelerator.accumulate(model):
                accelerator.backward(model(torch.ones(1, 1)).sum())
                optimizer.step()
                optimizer.zero_grad()

        torch.testing.assert_close(torch.stack(weights), torch.tensor([[[0.9]], [[0.8]]]))

    def test_optimizer_post_step_hook_skips_amp_overflow(self):
        # Exercise the AMP wrapper with a real CPU scaler without requiring a GPU.
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.ones(1))
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        optimizer = AcceleratedOptimizer(torch.optim.SGD([parameter], lr=0.1), scaler=scaler)
        weights, skipped = [], []
        optimizer.register_step_post_hook(lambda *args: weights.append(parameter.detach().clone()))

        for multiplier in [1.0, float("inf"), 1.0]:
            scaler.scale(parameter.sum() * multiplier).backward()
            optimizer.step()
            skipped.append(optimizer.step_was_skipped)
            optimizer.zero_grad()

        self.assertEqual(skipped, [False, True, False])
        torch.testing.assert_close(torch.stack(weights), torch.tensor([[0.9], [0.8]]))


@require_fp16
@require_non_cpu
class OptimizerTester(AccelerateTestCase):
    def test_accelerated_optimizer_step_was_skipped(self):
        model = torch.nn.Linear(5, 5)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        accelerator = Accelerator(mixed_precision="fp16")
        model, optimizer = accelerator.prepare(model, optimizer)

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            # Fake the gradients, as if there's no overflow
            p.grad.fill_(0.01)

        optimizer.step()
        assert optimizer.step_was_skipped is False

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            p.grad.fill_(0.01)
        # Manually set the gradients to be NaN, as if there's an overflow
        p.grad[0] = torch.tensor(float("nan"))

        optimizer.step()
        assert optimizer.step_was_skipped is True

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            p.grad.fill_(0.01)
        # Manually set the gradients to be NaN, as if there's an overflow
        p.grad[0] = torch.tensor(float("nan"))

        optimizer.step()
        assert optimizer.step_was_skipped is True

        loss = model(torch.randn(2, 5, device=accelerator.device)).sum()
        accelerator.backward(loss)
        for p in model.parameters():
            # Fake the gradients, as if there's no overflow
            p.grad.fill_(0.01)

        optimizer.step()
        assert optimizer.step_was_skipped is False
