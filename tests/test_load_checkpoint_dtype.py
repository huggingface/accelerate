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

import tempfile
from pathlib import Path
import unittest

import torch

from accelerate import load_checkpoint_and_dispatch
from accelerate.utils.modeling import load_checkpoint_in_model


class LoadCheckpointDtypeTest(unittest.TestCase):
    def test_load_checkpoint_and_dispatch_default_device_map_dtype(self):
        """Test that device_map=None respects dtype for fp32 checkpoints (Issue #4434)."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.bin"
            torch.save(torch.nn.Linear(4, 2).state_dict(), checkpoint)

            for device_map in (None, {"": "cpu"}):
                model = load_checkpoint_and_dispatch(
                    torch.nn.Linear(4, 2),
                    str(checkpoint),
                    device_map=device_map,
                    dtype=torch.float16,
                )
                self.assertEqual(model.weight.dtype, torch.float16)
                self.assertEqual(model.bias.dtype, torch.float16)
                output = model(torch.ones(1, 4, dtype=torch.float16))
                self.assertEqual(output.dtype, torch.float16)

    def test_load_checkpoint_and_dispatch_default_device_map_dtype_none(self):
        """Test that dtype=None preserves existing model dtype with device_map=None."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.bin"
            torch.save(torch.nn.Linear(4, 2).state_dict(), checkpoint)

            model = load_checkpoint_and_dispatch(
                torch.nn.Linear(4, 2),
                str(checkpoint),
                device_map=None,
                dtype=None,
            )
            self.assertEqual(model.weight.dtype, torch.float32)

    def test_load_checkpoint_and_dispatch_preserves_non_floating_buffers(self):
        """Test that int and bool buffers are not cast when dtype=torch.float16."""
        class ModelWithBuffers(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.register_buffer("int_buf", torch.tensor([1, 2, 3]))
                self.register_buffer("bool_buf", torch.tensor([True, False]))
                self.register_buffer("float_buf", torch.tensor([1.0, 2.0]))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_buffers.bin"
            torch.save(ModelWithBuffers().state_dict(), checkpoint)

            model = load_checkpoint_and_dispatch(
                ModelWithBuffers(),
                str(checkpoint),
                device_map=None,
                dtype=torch.float16,
            )
            self.assertEqual(model.linear.weight.dtype, torch.float16)
            self.assertEqual(model.float_buf.dtype, torch.float16)
            self.assertEqual(model.int_buf.dtype, torch.int64)
            self.assertEqual(model.bool_buf.dtype, torch.bool)

    def test_load_checkpoint_and_dispatch_keep_in_fp32_modules(self):
        """Test that keep_in_fp32_modules preserves float32 for specified modules."""
        class MultiModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 4)
                self.fc2 = torch.nn.Linear(4, 2)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_multi.bin"
            torch.save(MultiModule().state_dict(), checkpoint)

            model = load_checkpoint_and_dispatch(
                MultiModule(),
                str(checkpoint),
                device_map=None,
                dtype=torch.float16,
                keep_in_fp32_modules=["fc1"],
            )
            self.assertEqual(model.fc1.weight.dtype, torch.float32)
            self.assertEqual(model.fc2.weight.dtype, torch.float16)

    def test_load_checkpoint_and_dispatch_tied_weights_keep_in_fp32(self):
        """Test that tied parameters remain tied and remain fp32 if any alias matches keep_in_fp32_modules."""
        class TiedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.linear2.weight = self.linear1.weight

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_tied.bin"
            torch.save(TiedModule().state_dict(), checkpoint)

            # Test alias 1
            model1 = load_checkpoint_and_dispatch(
                TiedModule(),
                str(checkpoint),
                device_map=None,
                dtype=torch.float16,
                keep_in_fp32_modules=["linear1"],
            )
            self.assertEqual(model1.linear1.weight.dtype, torch.float32)
            self.assertEqual(model1.linear2.weight.dtype, torch.float32)
            self.assertTrue(model1.linear1.weight is model1.linear2.weight)

            # Test alias 2 (reverse registration order)
            model2 = load_checkpoint_and_dispatch(
                TiedModule(),
                str(checkpoint),
                device_map=None,
                dtype=torch.float16,
                keep_in_fp32_modules=["linear2"],
            )
            self.assertEqual(model2.linear1.weight.dtype, torch.float32)
            self.assertEqual(model2.linear2.weight.dtype, torch.float32)
            self.assertTrue(model2.linear1.weight is model2.linear2.weight)
