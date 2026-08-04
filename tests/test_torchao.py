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

import unittest

import torch

from accelerate.test_utils import require_torchao


@require_torchao
class TorchAOFilterTester(unittest.TestCase):
    """CPU-only checks on the module filters in `accelerate.utils.ao`.

    The layer swap itself needs no accelerator, so these do not go through the
    launcher like the torchao tests in `test_fp8.py`.
    """

    def test_convert_model_to_fp8_ao_skips_the_first_and_last_linear_layers(self):
        # convert_model_to_fp8_ao documents that it converts every nn.Linear "except the
        # first and last", and find_first_last_linear_layers exists because quantizing
        # those two destabilises training. The default module_filter_func was
        # filter_first_and_last_linear_layers, which re-derives the first and last linear
        # from the module it is handed. torchao hands it one candidate layer at a time, so
        # it always found ("", "") and filtered nothing, and every linear including the
        # first and last was converted.
        from torchao.float8.float8_linear import Float8Linear

        from accelerate.utils.ao import convert_model_to_fp8_ao

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_proj = torch.nn.Linear(32, 64)
                self.block = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU())
                self.lm_head = torch.nn.Linear(64, 32)

        model = ToyModel()
        convert_model_to_fp8_ao(model)

        assert not isinstance(model.embed_proj, Float8Linear)
        assert not isinstance(model.lm_head, Float8Linear)
        assert isinstance(model.block[0], Float8Linear)
