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

import sys
import unittest
from contextlib import ExitStack, contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from accelerate.utils import TERecipeKwargs, imports, transformer_engine


@contextmanager
def mock_modules(modules):
    # Restoring all of sys.modules can discard lazy imports while leaving their
    # native PyTorch operators registered, causing duplicate registration later.
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class TestTransformerEngineAvailability(unittest.TestCase):
    def test_mock_cleanup_preserves_unrelated_imports(self):
        names = ("_accelerate_te_existing", "_accelerate_te_missing", "_accelerate_te_lazy")
        existing, mocked, lazy = (ModuleType(name) for name in names)
        for name in names:
            self.addCleanup(sys.modules.pop, name, None)
        sys.modules[names[0]] = existing
        with mock_modules({names[0]: mocked, names[1]: mocked}):
            self.assertIs(sys.modules[names[0]], mocked)
            self.assertIs(sys.modules[names[1]], mocked)
            sys.modules[names[2]] = lazy
        self.assertIs(sys.modules[names[0]], existing)
        self.assertNotIn(names[1], sys.modules)
        self.assertIs(sys.modules[names[2]], lazy)

    def test_nvidia_minimum_version(self):
        with (
            patch.object(imports, "is_hpu_available", return_value=False),
            patch.object(imports, "_is_package_available", return_value=True),
        ):
            for version, expected in [("2.1.0", False), ("2.8.0", False), ("2.9.0", True), ("2.20.2", True)]:
                with self.subTest(version=version), patch("importlib.metadata.version", return_value=version):
                    self.assertEqual(imports.is_transformer_engine_available(), expected)
                    if not expected:
                        # Unsupported versions must never import TE's GPU extension.
                        self.assertFalse(imports.is_transformer_engine_mxfp8_available())
                        with self.assertRaisesRegex(ImportError, "supported TransformerEngine version"):
                            TERecipeKwargs()
        with (
            patch.object(imports, "is_hpu_available", return_value=False),
            patch.object(imports, "_is_package_available", return_value=False),
        ):
            self.assertFalse(imports.is_transformer_engine_available())


class TestTransformerEnginePublicAPI(unittest.TestCase):
    def setUp(self):
        self.calls = []

        # Strict signatures intentionally exclude deprecated fp8_autocast/fp8_recipe.
        @contextmanager
        def autocast(*, enabled, recipe):
            self.calls.append((enabled, recipe))
            yield

        def is_mxfp8_available(*, return_reason=False):
            return (True, "") if return_reason else True

        class Recipe:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class MXRecipe(Recipe):
            pass

        te = ModuleType("transformer_engine")
        pytorch = ModuleType("transformer_engine.pytorch")
        pytorch.autocast = autocast
        pytorch.is_mxfp8_available = MagicMock(side_effect=is_mxfp8_available)
        common = ModuleType("transformer_engine.common")
        recipe = ModuleType("transformer_engine.common.recipe")
        recipe.Format = SimpleNamespace(HYBRID=object(), E4M3=object())
        recipe.DelayedScaling = Recipe
        recipe.MXFP8BlockScaling = MXRecipe
        te.pytorch = pytorch
        te.common = common
        common.recipe = recipe
        self.te = pytorch
        self.recipe = recipe
        stack = ExitStack()
        self.addCleanup(stack.close)
        for context in [
            mock_modules(
                {
                    "transformer_engine": te,
                    "transformer_engine.pytorch": pytorch,
                    "transformer_engine.common": common,
                    "transformer_engine.common.recipe": recipe,
                },
            ),
            patch.object(transformer_engine, "is_hpu_available", return_value=False),
            patch.object(transformer_engine, "is_transformer_engine_available", return_value=True),
            patch("accelerate.utils.dataclasses.is_transformer_engine_available", return_value=True),
        ]:
            stack.enter_context(context)

    def test_train_eval_autocast_and_recipe(self):
        for use_during_eval in [False, True]:
            with self.subTest(use_during_eval=use_during_eval):
                model = torch.nn.Linear(16, 16)
                original_forward = model.forward
                handler = TERecipeKwargs(
                    margin=3,
                    fp8_format="E4M3",
                    amax_history_len=32,
                    amax_compute_algo="max",
                    use_autocast_during_eval=use_during_eval,
                )
                transformer_engine.apply_fp8_autowrap(model, handler)
                inputs = torch.randn(2, 16)
                for training in [True, False, True]:
                    model.train(training)
                    torch.testing.assert_close(model(inputs), original_forward(inputs))
                    enabled, recipe = self.calls[-1]
                    self.assertEqual(enabled, training or use_during_eval)
                    self.assertEqual(
                        recipe.kwargs,
                        {
                            "margin": 3,
                            "fp8_format": self.recipe.Format.E4M3,
                            "amax_history_len": 32,
                            "amax_compute_algo": "max",
                        },
                    )
                self.assertEqual(model.forward.__wrapped__, original_forward)
        self.te.is_mxfp8_available.assert_called_with(return_reason=True)

    def test_public_mxfp8_capability_check(self):
        with (
            patch.object(imports, "is_hpu_available", return_value=False),
            patch.object(imports, "is_transformer_engine_available", return_value=True),
        ):
            self.assertTrue(imports.is_transformer_engine_mxfp8_available())
        self.te.is_mxfp8_available.assert_called_once_with()

    def test_mxfp8_recipe_and_unavailable_reason(self):
        model = torch.nn.Linear(16, 16)
        handler = TERecipeKwargs(use_mxfp8_block_scaling=True)
        transformer_engine.apply_fp8_autowrap(model, handler)
        model(torch.randn(2, 16))
        self.assertIsInstance(self.calls[-1][1], self.recipe.MXFP8BlockScaling)
        self.te.is_mxfp8_available.return_value = (False, "unsupported GPU")
        self.te.is_mxfp8_available.side_effect = None
        with self.assertRaisesRegex(ValueError, "unsupported GPU"):
            transformer_engine.apply_fp8_autowrap(torch.nn.Linear(16, 16), handler)
