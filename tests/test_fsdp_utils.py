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
"""Unit tests for FSDP optimizer checkpointing helpers in `accelerate.utils.fsdp_utils`.

Regression tests for https://github.com/huggingface/accelerate/issues/4289: torch's
`torch.distributed.checkpoint` state-dict APIs call `optimizer.step()` on a dummy batch to
materialize empty optimizer state (`_init_optim_state`), and when handed accelerate's
`AcceleratedOptimizer` that dummy step is routed through the `GradScaler`
(`scaler.step()` + `scaler.update()`), mutating the restored scaler state as a side
effect of checkpointing. The DCP APIs must therefore receive the unwrapped optimizer.
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils.fsdp_utils import (
    _unwrap_optimizer_for_dcp,
    load_fsdp_optimizer,
    save_fsdp_optimizer,
)


def make_fsdp2_plugin():
    return SimpleNamespace(
        fsdp_version=2,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=SimpleNamespace(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=SimpleNamespace(),
    )


class FsdpOptimizerUnwrapTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # `AcceleratedOptimizer` requires the shared `AcceleratorState` to exist.
        Accelerator()

    def make_wrapped_optimizer(self):
        model = torch.nn.Linear(8, 8)
        inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
        wrapped = AcceleratedOptimizer(inner, device_placement=False)
        return model, inner, wrapped

    def test_unwrap_optimizer_for_dcp(self):
        _, inner, wrapped = self.make_wrapped_optimizer()
        self.assertIs(_unwrap_optimizer_for_dcp(wrapped), inner)
        self.assertIs(_unwrap_optimizer_for_dcp(inner), inner)
        double_wrapped = AcceleratedOptimizer(wrapped, device_placement=False)
        self.assertIs(_unwrap_optimizer_for_dcp(double_wrapped), inner)

    def test_load_fsdp_optimizer_passes_unwrapped_optimizer_to_dcp(self):
        model, inner, wrapped = self.make_wrapped_optimizer()
        accelerator = MagicMock()
        accelerator.process_index = 0

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("torch.distributed.checkpoint.state_dict.get_optimizer_state_dict") as mock_get,
            patch("torch.distributed.checkpoint.state_dict.set_optimizer_state_dict") as mock_set,
            patch("torch.distributed.checkpoint.load"),
            patch("torch.distributed.checkpoint.FileSystemReader"),
            patch("torch.distributed.checkpoint.default_planner.DefaultLoadPlanner"),
        ):
            mock_get.return_value = {"state": {}, "param_groups": []}
            load_fsdp_optimizer(make_fsdp2_plugin(), accelerator, wrapped, model, tmpdir, 0)

        self.assertEqual(mock_get.call_count, 1)
        self.assertIs(mock_get.call_args[0][1], inner)
        self.assertEqual(mock_set.call_count, 1)
        self.assertIs(mock_set.call_args[0][1], inner)

    def test_save_fsdp_optimizer_passes_unwrapped_optimizer_to_dcp(self):
        model, inner, wrapped = self.make_wrapped_optimizer()
        accelerator = MagicMock()
        accelerator.process_index = 0

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("torch.distributed.checkpoint.state_dict.get_optimizer_state_dict") as mock_get,
            patch("torch.distributed.checkpoint.save"),
            patch("torch.distributed.checkpoint.FileSystemWriter"),
            patch("torch.distributed.checkpoint.default_planner.DefaultSavePlanner"),
        ):
            save_fsdp_optimizer(make_fsdp2_plugin(), accelerator, wrapped, model, tmpdir, 0)

        self.assertEqual(mock_get.call_count, 1)
        self.assertIs(mock_get.call_args[0][1], inner)


if __name__ == "__main__":
    unittest.main()
