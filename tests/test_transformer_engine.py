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
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs, imports, transformer_engine
from accelerate.utils.dataclasses import FP8BackendType


class TestTransformerEngineAvailability(unittest.TestCase):
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

    def test_missing_package(self):
        with (
            patch.object(imports, "is_hpu_available", return_value=False),
            patch.object(imports, "_is_package_available", return_value=False),
        ):
            self.assertFalse(imports.is_transformer_engine_available())

    def test_intel_does_not_use_nvidia_version_requirement(self):
        with (
            patch.object(imports, "is_hpu_available", return_value=True),
            patch.object(imports, "_is_package_available", return_value=True) as available,
            patch.object(imports, "compare_versions") as compare,
        ):
            self.assertTrue(imports.is_transformer_engine_available())
            available.assert_called_once_with("intel_transformer_engine", "intel-transformer-engine")
            compare.assert_not_called()

    def test_recipe_reports_minimum_version(self):
        with patch("accelerate.utils.dataclasses.is_transformer_engine_available", return_value=False):
            with self.assertRaisesRegex(ImportError, r">= 2\.9\.0"):
                TERecipeKwargs()


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

        self.recipe_class = Recipe

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
        for patcher in [
            patch.dict(
                sys.modules,
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
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def test_intel_autocast_keeps_its_recipe_keyword(self):
        @contextmanager
        def fp8_autocast(*, enabled, fp8_recipe):
            self.calls.append((enabled, fp8_recipe))
            yield

        intel = ModuleType("intel_transformer_engine")
        intel.fp8_autocast = fp8_autocast
        model = torch.nn.Linear(16, 16)
        recipe = object()
        with (
            patch.dict(sys.modules, {"intel_transformer_engine": intel}),
            patch.object(transformer_engine, "is_hpu_available", return_value=True),
        ):
            forward = transformer_engine.contextual_fp8_autocast(model.forward, recipe)
            inputs = torch.randn(2, 16)
            for training in [True, False]:
                model.train(training)
                torch.testing.assert_close(forward(model, inputs), model(inputs))
                self.assertEqual(self.calls[-1], (training, recipe))

    def test_deepspeed_custom_recipe_reaches_te(self):
        # Exercise the real DeepSpeed preparation path through TE wrapping, stopping
        # before DeepSpeed engine initialization (which requires GPUs/process groups).
        class WrappingComplete(Exception):
            pass

        plugin = MagicMock()
        plugin.deepspeed_config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1}
        plugin.is_auto.return_value = False
        plugin.get_value.side_effect = plugin.deepspeed_config.get
        plugin.set_moe_leaf_modules.side_effect = WrappingComplete
        handler = TERecipeKwargs(margin=5, amax_history_len=32, use_autocast_during_eval=True)
        accelerator = object.__new__(Accelerator)
        accelerator.te_recipe_handler = handler
        # Distinct deprecated-slot settings catch incorrect handler precedence too.
        accelerator.fp8_recipe_handler = TERecipeKwargs(margin=1)
        model = torch.nn.Linear(16, 16)
        with (
            patch.dict(sys.modules, {"deepspeed": SimpleNamespace(initialize=MagicMock())}),
            patch.object(Accelerator, "parallelism_config", new_callable=PropertyMock, return_value=None),
            patch.object(Accelerator, "fp8_backend", new_callable=PropertyMock, return_value=FP8BackendType.TE),
            patch.object(Accelerator, "deepspeed_plugin", new_callable=PropertyMock, return_value=plugin),
            patch.object(Accelerator, "gradient_accumulation_steps", new_callable=PropertyMock, return_value=1),
            patch.object(Accelerator, "num_processes", new_callable=PropertyMock, return_value=1),
        ):
            with self.assertRaises(WrappingComplete):
                accelerator._prepare_deepspeed(model)
        model.eval()
        model(torch.randn(2, 16))
        enabled, recipe = self.calls[-1]
        self.assertTrue(enabled)
        self.assertEqual(recipe.kwargs["margin"], 5)
        self.assertEqual(recipe.kwargs["amax_history_len"], 32)
