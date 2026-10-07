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

import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
from parameterized import parameterized
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from accelerate.state import PartialState
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import is_torch_version
from accelerate.utils.modeling import load_checkpoint_in_model


def save_checkpoint(state_dict, directory, sharded=False, safetensors=False):
    extension = "safetensors" if safetensors else "bin"
    parts = [{key: value} for key, value in state_dict.items()] if sharded else [state_dict]
    weight_map = {}
    for index, part in enumerate(parts):
        filename = f"checkpoint_{index}.{extension}"
        path = os.path.join(directory, filename)
        if safetensors:
            save_file({key: value.clone() for key, value in part.items()}, path, metadata={"format": "pt"})
        else:
            torch.save(part, path)
        weight_map.update({key: filename for key in part})
    if not sharded:
        return path
    path = os.path.join(directory, "checkpoint.index.json")
    with open(path, "w") as handle:
        json.dump({"weight_map": weight_map}, handle)
    return path


def broadcast_checkpoint_worker(rank, directory):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for sharded in (False, True):
            for invalid_key in (None, "bias", "unexpected"):
                model = DistributedDataParallel(nn.Linear(2, 2))
                path = os.path.join(
                    directory,
                    f"case_{sharded}_{invalid_key}",
                    "checkpoint.index.json" if sharded else "checkpoint_0.bin",
                )
                try:
                    load_checkpoint_in_model(model, path, strict=True, broadcast_from_rank0=True)
                except RuntimeError as error:
                    if invalid_key is None or invalid_key not in str(error):
                        raise
                else:
                    if invalid_key is not None:
                        raise AssertionError(f"Rank {rank} accepted invalid key {invalid_key}")
                    torch.testing.assert_close(model.module.weight, torch.full((2, 2), 3.0))
                    torch.testing.assert_close(model.module.bias, torch.full((2,), 4.0))
                    with unittest.TestCase().assertNoLogs("accelerate.utils.modeling", level="WARNING"):
                        load_checkpoint_in_model(model, path, strict=False, broadcast_from_rank0=True)
    finally:
        dist.destroy_process_group()


class CheckpointKeyValidationTest(AccelerateTestCase):
    def setUp(self):
        PartialState(cpu=True)

    @parameterized.expand([(False, False), (True, False), (False, True), (True, True)])
    def test_strict_checkpoint_keys(self, sharded, safetensors):
        source = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).state_dict()
        for device_map in (None, {"": "cpu"}):
            for invalid_key in (None, "0.bias", "1.running_mean", "unexpected"):
                with self.subTest(device_map=device_map, invalid_key=invalid_key):
                    checkpoint = source.copy()
                    if invalid_key == "unexpected":
                        checkpoint[invalid_key] = torch.ones(1)
                    elif invalid_key is not None:
                        del checkpoint[invalid_key]
                    with tempfile.TemporaryDirectory() as directory:
                        path = save_checkpoint(checkpoint, directory, sharded, safetensors)
                        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                        if invalid_key is not None:
                            with self.assertRaisesRegex(RuntimeError, invalid_key):
                                load_checkpoint_in_model(model, path, device_map=device_map, strict=True)
                        else:
                            load_checkpoint_in_model(model, path, device_map=device_map, strict=True)
                            for key, value in model.state_dict().items():
                                torch.testing.assert_close(value, source[key])
                        # Partial checkpoints remain usable when the caller deliberately opts out.
                        load_checkpoint_in_model(model, path, device_map=device_map, strict=False)

    def test_strict_shards_legacy_loader(self):
        source = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(source.state_dict(), directory, sharded=True)
            model = nn.Linear(2, 2)
            with patch("accelerate.utils.modeling.is_torch_version", return_value=False):
                load_checkpoint_in_model(model, path, strict=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, source.state_dict()[key])

    def test_strict_tied_parameters(self):
        for device_map in (None, {"": "cpu"}):
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            model[1].weight = model[0].weight
            source = {key: value.clone() for key, value in model.state_dict().items()}
            with tempfile.TemporaryDirectory() as directory:
                path = save_checkpoint(source, directory, sharded=True, safetensors=True)
                load_checkpoint_in_model(model, path, device_map=device_map, strict=True)
                self.assertIs(model[0].weight, model[1].weight)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, source[key])
                # Strict key matching requires aliases too; non-strict loading remains available.
                del source["1.weight"]
                path = save_checkpoint(source, directory, sharded=True, safetensors=True)
                with self.assertRaisesRegex(RuntimeError, "1.weight"):
                    load_checkpoint_in_model(model, path, device_map=device_map, strict=True)

    def test_strict_quantization_auxiliary_keys(self):
        model = nn.Linear(2, 2)
        source = model.state_dict()
        source["quantization_SCB"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(source, directory)
            load_checkpoint_in_model(model, path, device_map={"": "cpu"}, strict=True)

    def test_strict_failure_cleans_temporary_offload(self):
        temporary_directories = []
        make_directory = tempfile.mkdtemp

        def record_directory(*args, **kwargs):
            directory = make_directory(*args, **kwargs)
            temporary_directories.append(directory)
            return directory

        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint({"weight": torch.ones(2, 2)}, directory)
            with patch("accelerate.utils.modeling.tempfile.mkdtemp", side_effect=record_directory):
                with self.assertRaisesRegex(RuntimeError, "bias"):
                    load_checkpoint_in_model(
                        nn.Linear(2, 2), path, device_map={"": "cpu"}, strict=True, offload_state_dict=True
                    )
        self.assertTrue(temporary_directories)
        self.assertTrue(all(not os.path.exists(directory) for directory in temporary_directories))

    @unittest.skipUnless(is_torch_version(">=", "2.2.0"), "Requires distributed checkpoint state dict support")
    def test_strict_compiled_model_canonical_keys(self):
        source = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = save_checkpoint(source.state_dict(), directory, sharded=True)
            for strict in (False, True):
                model = torch.compile(nn.Linear(2, 2), backend="eager")
                with self.assertNoLogs("accelerate.utils.modeling", level="WARNING"):
                    load_checkpoint_in_model(model, path, strict=strict)
                torch.testing.assert_close(model.weight, source.weight)
                torch.testing.assert_close(model.bias, source.bias)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available() and is_torch_version(">=", "2.4.0"),
        "Requires Gloo and rank-zero checkpoint broadcasting",
    )
    def test_strict_broadcast_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            for sharded in (False, True):
                for invalid_key in (None, "bias", "unexpected"):
                    case_directory = os.path.join(directory, f"case_{sharded}_{invalid_key}")
                    os.mkdir(case_directory)
                    state_dict = {"weight": torch.full((2, 2), 3.0)}
                    if invalid_key != "bias":
                        state_dict["bias"] = torch.full((2,), 4.0)
                    if invalid_key == "unexpected":
                        state_dict[invalid_key] = torch.ones(1)
                    save_checkpoint(state_dict, case_directory, sharded=sharded)
            # macOS does not resolve the local hostname on every CI runner.
            environment = {"GLOO_SOCKET_IFNAME": "lo0"} if sys.platform == "darwin" else {}
            with patch.dict(os.environ, environment):
                torch.multiprocessing.spawn(broadcast_checkpoint_worker, args=(directory,), nprocs=2, join=True)
