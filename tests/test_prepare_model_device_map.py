# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import unittest
from unittest.mock import patch

import torch

from accelerate import Accelerator
from accelerate.test_utils import assert_exception, require_non_torch_xla


class PrepareModelDeviceMapTester(unittest.TestCase):
    @require_non_torch_xla
    def test_prepare_model_8bit_cpu_offload_raises_valueerror_not_typeerror(self):
        class ModelForTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.l(x)

        accelerator = Accelerator()
        model = ModelForTest()

        # Trigger the 8-bit/4-bit + hf_device_map code path.
        model.is_loaded_in_8bit = True
        model.hf_device_map = {"": "cpu"}

        with (
            patch("accelerate.accelerator.is_bitsandbytes_multi_backend_available", return_value=False),
            patch("accelerate.accelerator.is_xpu_available", return_value=False),
        ):
            with assert_exception(ValueError, "CPU or disk offload"):
                accelerator.prepare_model(model)
