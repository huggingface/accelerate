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
from itertools import chain
from unittest import skipUnless

import torch
from parameterized import parameterized

from accelerate import Accelerator
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import DistributedType, is_torch_xla_available


class MultipleModelParameterTest(AccelerateTestCase):
    @parameterized.expand([(nested, drop_refs) for nested in (False, True) for drop_refs in (False, True)])
    def test_single_model_names_are_unchanged(self, nested, drop_refs):
        accelerator = Accelerator()
        model = torch.nn.Sequential(torch.nn.Linear(3, 2)) if nested else torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        actual = accelerator._get_named_parameters(optimizer, model, object(), drop_refs=drop_refs)
        expected = dict(model.named_parameters())
        self.assertEqual(list(actual), list(expected))
        for name, parameter in expected.items():
            if drop_refs:
                self.assertEqual(actual[name], parameter.data_ptr())
            else:
                self.assertIs(actual[name], parameter)

    @parameterized.expand(
        [
            (count, drop_refs, interleaved)
            for count in (2, 3)
            for drop_refs in (False, True)
            for interleaved in (False, True)
        ]
    )
    def test_same_names_in_multiple_models_are_preserved(self, count, drop_refs, interleaved):
        accelerator = Accelerator()
        models = [torch.nn.Sequential(torch.nn.Linear(3, 2)) for _ in range(count)]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in models]
        args = list(chain.from_iterable(zip(models, optimizers))) if interleaved else [*models, *optimizers]

        actual = accelerator._get_named_parameters(*args, drop_refs=drop_refs)
        expected = list(chain.from_iterable(model.parameters() for model in models))
        self.assertEqual(len(actual), len(expected))
        if drop_refs:
            self.assertEqual(list(actual.values()), [parameter.data_ptr() for parameter in expected])
        else:
            for actual_parameter, expected_parameter in zip(actual.values(), expected):
                self.assertIs(actual_parameter, expected_parameter)

    @parameterized.expand([(False,), (True,)])
    def test_non_models_have_no_parameters(self, drop_refs):
        accelerator = Accelerator()
        self.assertEqual(accelerator._get_named_parameters(object(), None, drop_refs=drop_refs), {})

    @parameterized.expand([(False,), (True,)])
    def test_empty_models_do_not_hide_other_parameters(self, drop_refs):
        accelerator = Accelerator()
        models = [torch.nn.Linear(3, 2), torch.nn.Module(), torch.nn.Linear(3, 2)]
        actual = accelerator._get_named_parameters(*models, drop_refs=drop_refs)
        expected = list(chain.from_iterable(model.parameters() for model in models))
        self.assertEqual(len(actual), len(expected))
        for actual_parameter, expected_parameter in zip(actual.values(), expected):
            if drop_refs:
                self.assertEqual(actual_parameter, expected_parameter.data_ptr())
            else:
                self.assertIs(actual_parameter, expected_parameter)


@skipUnless(is_torch_xla_available(), "requires the native PyTorch/XLA runtime")
class XLAMultipleModelOptimizerTest(AccelerateTestCase):
    @parameterized.expand(
        [
            (optimizer_name, count, joint, interleaved)
            for optimizer_name in ("sgd", "adamw")
            for count, joint, interleaved in (
                (1, False, False),
                (2, False, False),
                (2, False, True),
                (3, False, True),
                (2, True, False),
                (3, True, False),
            )
        ]
    )
    def test_prepare_preserves_each_optimizer_training_trajectory(self, optimizer_name, count, joint, interleaved):
        import torch_xla

        accelerator = Accelerator()
        self.assertEqual(accelerator.distributed_type, DistributedType.XLA)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            models = [
                torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
                for _ in range(count)
            ]
        references = [copy.deepcopy(model).to(accelerator.device) for model in models]
        initial_parameters = [[parameter.detach().clone() for parameter in model.parameters()] for model in models]

        def make_optimizer(parameters):
            if optimizer_name == "sgd":
                return torch.optim.SGD(parameters, lr=0.05, momentum=0.8, weight_decay=0.01)
            return torch.optim.AdamW(parameters, lr=0.02, betas=(0.8, 0.9), weight_decay=0.01)

        # The references follow native XLA's ordering: move the models before creating their optimizers.
        if joint:
            optimizers = [make_optimizer(chain.from_iterable(model.parameters() for model in models))]
            reference_optimizers = [make_optimizer(chain.from_iterable(model.parameters() for model in references))]
        else:
            optimizers = [make_optimizer(model.parameters()) for model in models]
            reference_optimizers = [make_optimizer(model.parameters()) for model in references]

        if interleaved:
            prepared = accelerator.prepare(*chain.from_iterable(zip(models, optimizers)))
            models, optimizers = list(prepared[::2]), list(prepared[1::2])
        else:
            prepared = accelerator.prepare(*models, *optimizers)
            models, optimizers = list(prepared[:count]), list(prepared[count:])

        inputs = torch.tensor([[0.5, 1.0, -0.5], [1.0, -0.5, 0.25]], device=accelerator.device)
        targets = torch.tensor([[0.25, -0.75], [1.0, 0.5]], device=accelerator.device)
        for step in range(3):
            for optimizer in [*optimizers, *reference_optimizers]:
                optimizer.zero_grad()
            losses = [
                torch.nn.functional.mse_loss(model(inputs + 0.1 * step), targets + 0.2 * index)
                for index, model in enumerate(models)
            ]
            reference_losses = [
                torch.nn.functional.mse_loss(model(inputs + 0.1 * step), targets + 0.2 * index)
                for index, model in enumerate(references)
            ]
            accelerator.backward(sum(losses))
            sum(reference_losses).backward()

            for loss, reference_loss in zip(losses, reference_losses):
                torch.testing.assert_close(loss.cpu(), reference_loss.cpu(), rtol=0, atol=0)
            for model, reference in zip(models, references):
                for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad.cpu(), reference_parameter.grad.cpu(), rtol=0, atol=0)

            for optimizer in optimizers:
                optimizer.step()
            for optimizer in reference_optimizers:
                optimizer.step()
            torch_xla.sync()

            for model, reference in zip(models, references):
                for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.cpu(), reference_parameter.cpu(), rtol=0, atol=0)
            for optimizer, reference_optimizer in zip(optimizers, reference_optimizers):
                actual_state = optimizer.state_dict()
                expected_state = reference_optimizer.state_dict()
                self.assertEqual(actual_state["param_groups"], expected_state["param_groups"])
                self.assertEqual(actual_state["state"].keys(), expected_state["state"].keys())
                for key, expected in expected_state["state"].items():
                    self.assertEqual(actual_state["state"][key].keys(), expected.keys())
                    for name, value in expected.items():
                        actual = actual_state["state"][key][name]
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(actual.cpu(), value.cpu(), rtol=0, atol=0)
                        else:
                            self.assertEqual(actual, value)

        expected_groups = (
            [list(chain.from_iterable(model.parameters() for model in models))]
            if joint
            else [list(model.parameters()) for model in models]
        )
        for optimizer, expected in zip(optimizers, expected_groups):
            for parameter, expected_parameter in zip(optimizer.param_groups[0]["params"], expected):
                self.assertIs(parameter, expected_parameter)
        for model, initial in zip(models, initial_parameters):
            self.assertTrue(
                any(not torch.equal(parameter.cpu(), old) for parameter, old in zip(model.parameters(), initial))
            )
