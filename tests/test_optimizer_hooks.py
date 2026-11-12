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

import io
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch
from parameterized import parameterized

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.test_utils import require_cpu
from accelerate.test_utils.testing import AccelerateTestCase


@require_cpu
class OptimizerHooksTester(AccelerateTestCase):
    @parameterized.expand([(name, accumulation) for name in ("SGD", "Adam") for accumulation in (1, 2, 3)])
    def test_step_hooks_match_native_updates(self, name, accumulation):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = torch.nn.Linear(2, 1)
        reference = deepcopy(model)
        optimizer_class = getattr(torch.optim, name)
        optimizer = accelerator.prepare(optimizer_class(model.parameters(), lr=0.1))
        reference_optimizer = optimizer_class(reference.parameters(), lr=0.1)
        events = []

        def pre_hook(owner, args, kwargs):
            self.assertIs(owner, optimizer.optimizer)
            events.append("pre")
            for group in owner.param_groups:
                for parameter in group["params"]:
                    parameter.grad.mul_(0.5)

        def post_hook(owner, args, kwargs):
            self.assertIs(owner, optimizer.optimizer)
            events.append("post")

        pre_handle = optimizer.register_step_pre_hook(pre_hook)
        post_handle = optimizer.register_step_post_hook(post_hook)
        for index in range(2 * accumulation):
            inputs = torch.tensor([[1.0, index + 1.0]])
            with accelerator.accumulate(model):
                accelerator.backward(model(inputs).square().mean())
                (reference(inputs).square().mean() / accumulation).backward()
                optimizer.step()
                optimizer.zero_grad()
                if (index + 1) % accumulation == 0:
                    for parameter in reference.parameters():
                        parameter.grad.mul_(0.5)
                    reference_optimizer.step()
                    reference_optimizer.zero_grad()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected)
        self.assertEqual(events, ["pre", "post"] * 2)

        pre_handle.remove()
        post_handle.remove()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertEqual(events, ["pre", "post"] * 2)

    def test_step_pre_hook_can_replace_closure(self):
        accelerator = Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = accelerator.prepare(torch.optim.SGD([parameter], lr=0.1))
        calls = []

        def original_closure():
            self.fail("The step pre-hook should replace this closure")

        def replacement_closure():
            optimizer.zero_grad()
            loss = (parameter - 0.5).square()
            loss.backward()
            calls.append("closure")
            return loss

        def pre_hook(owner, args, kwargs):
            self.assertIs(args[0], optimizer.optimizer)
            return (args[0], replacement_closure), kwargs

        optimizer.register_step_pre_hook(pre_hook)
        optimizer.step(original_closure)
        self.assertEqual(calls, ["closure"])
        torch.testing.assert_close(parameter, torch.tensor(1.7))

    @parameterized.expand([(phase, prepend) for phase in ("pre", "post") for prepend in (False, True)])
    def test_state_dict_hook_order_and_removal(self, phase, prepend):
        accelerator = Accelerator(cpu=True)
        optimizer = accelerator.prepare(torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1))
        events = []

        def existing(owner, *args):
            events.append("existing")

        def added(owner, *args):
            self.assertIs(owner, optimizer.optimizer)
            events.append("added")
            if phase == "post":
                return {**args[0], "hook_value": 23}

        getattr(optimizer.optimizer, f"register_state_dict_{phase}_hook")(existing)
        handle = getattr(optimizer, f"register_state_dict_{phase}_hook")(added, prepend=prepend)
        state = optimizer.state_dict()
        self.assertEqual(events, ["added", "existing"] if prepend else ["existing", "added"])
        self.assertEqual(state.get("hook_value"), 23 if phase == "post" else None)
        events.clear()
        handle.remove()
        self.assertNotIn("hook_value", optimizer.state_dict())
        self.assertEqual(events, ["existing"])

    @parameterized.expand([(phase, prepend) for phase in ("pre", "post") for prepend in (False, True)])
    def test_load_state_dict_hook_order_and_removal(self, phase, prepend):
        accelerator = Accelerator(cpu=True)
        optimizer = accelerator.prepare(torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1))
        state = optimizer.state_dict()
        events = []

        def existing(owner, *args):
            events.append("existing")

        def added(owner, *args):
            self.assertIs(owner, optimizer.optimizer)
            events.append("added")
            if phase == "pre":
                state = deepcopy(args[0])
                state["param_groups"][0]["lr"] = 0.2
                return state

        getattr(optimizer.optimizer, f"register_load_state_dict_{phase}_hook")(existing)
        handle = getattr(optimizer, f"register_load_state_dict_{phase}_hook")(added, prepend=prepend)
        optimizer.load_state_dict(state)
        self.assertEqual(events, ["added", "existing"] if prepend else ["existing", "added"])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.2 if phase == "pre" else 0.1)
        events.clear()
        handle.remove()
        optimizer.load_state_dict(state)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(events, ["existing"])

    def test_state_dict_hooks_roundtrip_real_adam_training(self):
        accelerator = Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = accelerator.prepare(torch.optim.Adam([parameter], lr=0.05))
        parameter.square().backward()
        optimizer.step()
        optimizer.zero_grad()
        optimizer.register_state_dict_post_hook(lambda owner, state: {"saved_optimizer": state})
        checkpoint = io.BytesIO()
        torch.save(optimizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)

        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = accelerator.prepare(torch.optim.Adam([resumed_parameter], lr=0.9))
        resumed.register_load_state_dict_pre_hook(lambda owner, state: state["saved_optimizer"])
        resumed.load_state_dict(saved)
        for _ in range(2):
            parameter.square().backward()
            resumed_parameter.square().backward()
            optimizer.step()
            resumed.step()
            optimizer.zero_grad()
            resumed.zero_grad()
            torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
            for key in optimizer.state[parameter]:
                torch.testing.assert_close(optimizer.state[parameter][key], resumed.state[resumed_parameter][key])

    def test_state_dict_hooks_work_with_accelerator_checkpoint(self):
        accelerator = Accelerator(cpu=True)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = torch.nn.Linear(2, 1)
        model, optimizer = accelerator.prepare(model, torch.optim.Adam(model.parameters(), lr=0.05))
        accelerator.backward(model(torch.ones(1, 2)).square().mean())
        optimizer.step()
        optimizer.zero_grad()
        expected_model = deepcopy(model.state_dict())
        expected_optimizer = deepcopy(optimizer.state_dict())
        events = []

        def save_hook(owner, state):
            events.append("save")
            return {"saved_optimizer": state}

        def load_hook(owner, state):
            events.append("load")
            return state["saved_optimizer"]

        optimizer.register_state_dict_post_hook(save_hook)
        optimizer.register_load_state_dict_pre_hook(load_hook)
        with TemporaryDirectory() as directory:
            accelerator.save_state(directory)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            optimizer.state.clear()
            optimizer.param_groups[0]["lr"] = 0.9
            accelerator.load_state(directory)
        self.assertEqual(events, ["save", "load"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
        actual_optimizer = optimizer.optimizer.state_dict()["saved_optimizer"]
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        for key, state in expected_optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(actual_optimizer["state"][key][name], value, rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch.amp, "GradScaler"), "CPU GradScaler requires a recent PyTorch")
    def test_step_hooks_skip_real_cpu_grad_scaler_overflow(self):
        Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        optimizer = AcceleratedOptimizer(torch.optim.SGD([parameter], lr=0.1), scaler=scaler)
        events = []
        optimizer.register_step_pre_hook(lambda *args: events.append("pre"))
        optimizer.register_step_post_hook(lambda *args: events.append("post"))
        for overflow in (False, True, False):
            optimizer.zero_grad()
            scaler.scale(parameter.square()).backward()
            if overflow:
                parameter.grad.fill_(float("inf"))
            optimizer.step()
            self.assertEqual(optimizer.step_was_skipped, overflow)
        torch.testing.assert_close(parameter, torch.tensor(0.64))
        self.assertEqual(events, ["pre", "post"] * 2)

    def test_existing_step_hook_remains_registered_after_prepare(self):
        accelerator = Accelerator(cpu=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        native = torch.optim.SGD([parameter], lr=0.1)
        events = []
        native.register_step_post_hook(lambda *args: events.append("post"))
        optimizer = accelerator.prepare(native)
        parameter.square().backward()
        optimizer.step()
        self.assertEqual(events, ["post"])
        torch.testing.assert_close(parameter, torch.tensor(0.8))
