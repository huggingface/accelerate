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

import inspect
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers.trainer_utils import set_seed

from accelerate.accelerator import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils.testing import AccelerateTestCase, require_cuda, require_deepspeed
from accelerate.test_utils.training import RegressionDataset
from accelerate.utils import patch_environment
from accelerate.utils.dataclasses import DeepSpeedPlugin


set_seed(42)

GPT2_TINY = "sshleifer/tiny-gpt2"
ZERO2 = "zero2"
ZERO3 = "zero3"
FP16 = "fp16"


@require_deepspeed
@require_cuda
class DeepSpeedGradientAccumulationTest(AccelerateTestCase):
    def setUp(self):
        super().setUp()

        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self.test_file_dir_str = str(path.parents[0])

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        # Load config files
        with open(self.ds_config_file[ZERO2], encoding="utf-8") as f:
            config_zero2 = json.load(f)
        with open(self.ds_config_file[ZERO3], encoding="utf-8") as f:
            config_zero3 = json.load(f)
            config_zero3["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False

        self.ds_config_dict = dict(zero2=config_zero2, zero3=config_zero3)

        self.dist_env = dict(
            ACCELERATE_USE_DEEPSPEED="true",
            MASTER_ADDR="localhost",
            MASTER_PORT="10999",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def get_config_dict(self, stage):
        # As some tests modify the dict, always make a copy
        return deepcopy(self.ds_config_dict[stage])

    def test_gradient_accumulation_boundary_integration(self):
        """Test that gradient accumulation boundaries are automatically handled by DeepSpeed integration."""
        gradient_accumulation_steps = 4

        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=False,
            zero3_init_flag=False,
        )

        with patch_environment(**self.dist_env):
            accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_plugin)

            # Setup simple training components
            train_set = RegressionDataset(length=80)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
            model = AutoModel.from_pretrained(GPT2_TINY)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            model.train()

            # Test gradient accumulation with accumulate context manager
            batch_data = next(iter(train_dataloader))
            # Create proper input format for GPT2 model (RegressionDataset returns {"x": scalar, "y": scalar})
            # We need to create dummy input_ids for the GPT2 model
            batch_size = len(batch_data["x"]) if isinstance(batch_data["x"], torch.Tensor) else 1
            if isinstance(batch_data["x"], torch.Tensor):
                batch_size = batch_data["x"].shape[0]
            else:
                batch_size = 1

            # Create dummy input_ids for GPT2 model and move to same device as model
            device = next(model.parameters()).device
            input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)  # batch_size x sequence_length
            inputs = {"input_ids": input_ids}

            # Track sync_gradients values to verify correct gradient accumulation behavior
            sync_values = []

            # Simulate gradient accumulation steps
            for micro_step in range(gradient_accumulation_steps):
                with accelerator.accumulate(model):
                    sync_values.append(accelerator.sync_gradients)
                    outputs = model(**inputs)
                    # Use the last hidden state and create a simple loss
                    prediction = outputs.last_hidden_state.mean()
                    loss = prediction.sum()  # Simple scalar loss

                    # This should automatically handle gradient accumulation boundaries
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        optimizer.step()
                        optimizer.zero_grad()

            # Verify gradient accumulation pattern was correct
            # Should be False for first 3 steps, True for the last step
            expected_sync = [False, False, False, True]
            self.assertEqual(sync_values, expected_sync)

            # Reset step counter for accelerator
            accelerator.step = 0

    def test_clip_grad_norm_returns_deepspeed_grad_norm(self):
        """Test that clip_grad_norm_ works with DeepSpeed and returns gradient norm when available."""
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=False,
            zero3_init_flag=False,
        )

        with patch_environment(**self.dist_env):
            accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_plugin)

            # Setup simple model
            model = AutoModel.from_pretrained(GPT2_TINY)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            # Create a simple dataloader for prepare to work
            train_set = RegressionDataset(length=16)
            train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)

            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            # Perform a forward and backward pass to generate gradients
            batch_data = next(iter(train_dataloader))
            batch_size = len(batch_data["x"]) if isinstance(batch_data["x"], torch.Tensor) else 1
            if isinstance(batch_data["x"], torch.Tensor):
                batch_size = batch_data["x"].shape[0]
            else:
                batch_size = 1

            # Create dummy input_ids for GPT2 model and move to same device as model
            device = next(model.parameters()).device
            input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
            inputs = {"input_ids": input_ids}

            # Forward pass
            outputs = model(**inputs)
            prediction = outputs.last_hidden_state.mean()
            loss = prediction.sum()

            # Backward pass to generate gradients
            accelerator.backward(loss)

            # Test that gradient clipping works and returns a value
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # After backward pass, we should get a valid gradient norm (either from DeepSpeed or fallback)
            self.assertIsInstance(grad_norm, (int, float, type(None)))
            if grad_norm is not None:
                self.assertGreaterEqual(grad_norm, 0.0)

    def test_accelerator_backward_passes_sync_gradients(self):
        """Test that Accelerator.backward() passes sync_gradients to DeepSpeed wrapper."""
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=2,
            gradient_clipping=1.0,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_save_16bit_model=False,
            zero3_init_flag=False,
        )

        with patch_environment(**self.dist_env):
            accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_plugin)

            # Setup simple model and data
            model = AutoModel.from_pretrained(GPT2_TINY)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            train_set = RegressionDataset(length=16)
            train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)

            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            # Track sync_gradients values during backward calls
            sync_values = []

            # Test two gradient accumulation steps
            batch_data = next(iter(train_dataloader))
            # Create proper input format for GPT2 model
            batch_size = len(batch_data["x"]) if isinstance(batch_data["x"], torch.Tensor) else 1
            if isinstance(batch_data["x"], torch.Tensor):
                batch_size = batch_data["x"].shape[0]
            else:
                batch_size = 1

            # Create dummy input_ids for GPT2 model and move to same device as model
            device = next(model.parameters()).device
            input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
            inputs = {"input_ids": input_ids}

            # First step - should have sync_gradients=False
            with accelerator.accumulate(model):
                sync_values.append(accelerator.sync_gradients)
                outputs = model(**inputs)
                prediction = outputs.last_hidden_state.mean()
                loss = prediction  # Simple loss
                accelerator.backward(loss)

            # Second step - should have sync_gradients=True
            with accelerator.accumulate(model):
                sync_values.append(accelerator.sync_gradients)
                outputs = model(**inputs)
                prediction = outputs.last_hidden_state.mean()
                loss = prediction  # Simple loss
                accelerator.backward(loss)

            # Verify sync_gradients pattern was correct
            self.assertEqual(len(sync_values), 2)
            self.assertFalse(sync_values[0])  # First step: not syncing
            self.assertTrue(sync_values[1])  # Second step: syncing

    def test_gradient_accumulation_with_different_steps(self):
        """Test gradient accumulation with different accumulation step values."""
        for gradient_accumulation_steps in [1, 2, 4, 8]:
            with self.subTest(gradient_accumulation_steps=gradient_accumulation_steps):
                AcceleratorState._reset_state()

                deepspeed_plugin = DeepSpeedPlugin(
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    gradient_clipping=1.0,
                    zero_stage=2,
                    offload_optimizer_device="cpu",
                    offload_param_device="cpu",
                    zero3_save_16bit_model=False,
                    zero3_init_flag=False,
                )

                with patch_environment(**self.dist_env):
                    accelerator = Accelerator(mixed_precision="fp16", deepspeed_plugin=deepspeed_plugin)

                    # Ensure the step counter starts at 0 for predictable behavior
                    accelerator.step = 0

                    # Track sync_gradients values
                    sync_values = []

                    # Simulate the gradient accumulation loop
                    for step in range(gradient_accumulation_steps):
                        with accelerator.accumulate(None):  # Don't need actual model for this test
                            sync_values.append(accelerator.sync_gradients)

                    # Verify sync pattern
                    expected_sync = [False] * (gradient_accumulation_steps - 1) + [True]
                    if gradient_accumulation_steps == 1:
                        expected_sync = [True]  # Special case: always sync when steps=1

                    self.assertEqual(
                        sync_values,
                        expected_sync,
                        f"Failed for gradient_accumulation_steps={gradient_accumulation_steps}",
                    )
