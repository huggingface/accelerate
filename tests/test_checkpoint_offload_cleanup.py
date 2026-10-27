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
import tempfile
import unittest
from unittest.mock import patch

import torch
from parameterized import parameterized
from safetensors import SafetensorError
from safetensors.torch import save_file

from accelerate import load_checkpoint_and_dispatch
from accelerate.utils.modeling import load_checkpoint_in_model
from accelerate.utils.offload import offload_weight


class CheckpointOffloadCleanupTest(unittest.TestCase):
    @parameterized.expand(
        [
            (failure, safe)
            for failure in ("missing_shard", "corrupt_shard", "offload_write", "restore", None)
            for safe in (False, True)
        ]
    )
    def test_temporary_state_dict_cleanup(self, failure, safe):
        source = torch.nn.Sequential(*(torch.nn.Linear(2, 2, bias=False) for _ in range(3)))
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters()):
                parameter.fill_(index + 1)
        state_dict = source.state_dict()
        if failure == "restore":
            state_dict["2.weight"] = torch.ones(3, 2)

        with tempfile.TemporaryDirectory() as directory:
            extension = "safetensors" if safe else "bin"
            first = os.path.join(directory, f"1.{extension}")
            second = os.path.join(directory, f"2.{extension}")
            for path, keys in ((first, ["0.weight", "1.weight"]), (second, ["2.weight"])):
                if path == second and failure == "missing_shard":
                    continue
                if path == second and failure == "corrupt_shard":
                    with open(path, "wb"):
                        pass
                    continue
                shard = {key: state_dict[key] for key in keys}
                if safe:
                    save_file(shard, path, metadata={"format": "pt"})
                else:
                    torch.save(shard, path)
            checkpoint = os.path.join(directory, "model.index.json")
            with open(checkpoint, "w") as handle:
                json.dump(
                    {
                        "weight_map": {
                            key: os.path.basename(first if key != "2.weight" else second) for key in state_dict
                        }
                    },
                    handle,
                )

            caller_folder = os.path.join(directory, "caller_offload")
            os.mkdir(caller_folder)
            marker = os.path.join(caller_folder, "keep.txt")
            with open(marker, "w") as handle:
                handle.write("caller-owned data")
            temporary_directories = []
            temporary_weights = []
            make_directory = tempfile.mkdtemp
            write_error = OSError("temporary weight write failed")

            def make_offload_directory(*args, **kwargs):
                kwargs["dir"] = directory
                result = make_directory(*args, **kwargs)
                temporary_directories.append(result)
                return result

            def write_offload_weight(weight, name, folder, index=None):
                offload_weight(weight, name, folder, index=index)
                if folder in temporary_directories:
                    path = os.path.join(folder, f"{name}.dat")
                    self.assertEqual(os.path.getsize(path), weight.numel() * weight.element_size())
                    temporary_weights.append(path)
                    if failure == "offload_write":
                        raise write_error

            model = torch.nn.Sequential(*(torch.nn.Linear(2, 2, bias=False) for _ in range(3)))
            kwargs = {
                "device_map": {"0": "cpu", "1": "disk", "2": "cpu"},
                "offload_folder": caller_folder,
                "offload_state_dict": True,
            }
            with (
                patch("accelerate.utils.modeling.tempfile.mkdtemp", side_effect=make_offload_directory),
                patch("accelerate.utils.modeling.offload_weight", new=write_offload_weight),
            ):
                if failure is None:
                    model = load_checkpoint_and_dispatch(model, checkpoint, **kwargs)
                    sample = torch.ones(1, 2)
                    with torch.no_grad():
                        torch.testing.assert_close(model(sample), source(sample))
                    torch.testing.assert_close(model[0].weight, source[0].weight)
                    torch.testing.assert_close(model[2].weight, source[2].weight)
                else:
                    error_type = {
                        "missing_shard": FileNotFoundError,
                        "corrupt_shard": SafetensorError if safe else EOFError,
                        "offload_write": OSError,
                        "restore": ValueError,
                    }[failure]
                    with self.assertRaises(error_type) as error:
                        load_checkpoint_in_model(model, checkpoint, **kwargs)
                    if failure == "offload_write":
                        self.assertIs(error.exception, write_error)
                    elif failure == "restore":
                        self.assertIn("Trying to set a tensor of shape", str(error.exception))

            self.assertTrue(temporary_directories)
            self.assertTrue(temporary_weights, "The test must actually offload weights before returning or raising")
            self.assertTrue(all(not os.path.exists(path) for path in temporary_directories))
            with open(marker) as handle:
                self.assertEqual(handle.read(), "caller-owned data")
            if failure != "offload_write":
                self.assertTrue(os.path.isfile(os.path.join(caller_folder, "1.weight.dat")))


if __name__ == "__main__":
    unittest.main()
