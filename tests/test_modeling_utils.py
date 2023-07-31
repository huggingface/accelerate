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

import json
import os
import tempfile
import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from accelerate.test_utils import require_cuda, require_huggingface_suite, require_multi_gpu, require_safetensors
from accelerate.utils.modeling import (
    check_device_map,
    clean_device_map,
    compute_module_sizes,
    convert_file_size_to_int,
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    load_checkpoint_in_model,
    load_state_dict,
    named_module_tensors,
    retie_parameters,
    set_module_tensor_to_device,
)


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


def sequential_model(num_layers):
    layers = OrderedDict([(f"linear{i}", nn.Linear(1000, 1000)) for i in range(1, num_layers + 1)])
    return nn.Sequential(layers)


class ModelingUtilsTester(unittest.TestCase):
    def check_set_module_tensor_for_device(self, model, device1, device2):
        self.assertEqual(model.linear1.weight.device, torch.device(device1))

        with self.subTest("Access by submodule and direct name for a parameter"):
            set_module_tensor_to_device(model.linear1, "weight", device2)
            self.assertEqual(model.linear1.weight.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model.linear1, "weight", device1)

                set_module_tensor_to_device(model.linear1, "weight", device1, value=torch.randn(4, 3))
            else:
                set_module_tensor_to_device(model.linear1, "weight", device1)
            self.assertEqual(model.linear1.weight.device, torch.device(device1))

        with self.subTest("Access by module and full name for a parameter"):
            set_module_tensor_to_device(model, "linear1.weight", device2)
            self.assertEqual(model.linear1.weight.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model, "linear1.weight", device1)
                set_module_tensor_to_device(model, "linear1.weight", device1, value=torch.randn(4, 3))
            else:
                set_module_tensor_to_device(model, "linear1.weight", device1)
            self.assertEqual(model.linear1.weight.device, torch.device(device1))

        self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

        with self.subTest("Access by submodule and direct name for a buffer"):
            set_module_tensor_to_device(model.batchnorm, "running_mean", device2)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on device1
                    set_module_tensor_to_device(model.batchnorm, "running_mean", device1)
                set_module_tensor_to_device(model.batchnorm, "running_mean", device1, value=torch.randn(4))
            else:
                set_module_tensor_to_device(model.batchnorm, "running_mean", device1)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

        with self.subTest("Access by module and full name for a parameter"):
            set_module_tensor_to_device(model, "batchnorm.running_mean", device2)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device2))

            if torch.device(device2) == torch.device("meta"):
                with self.assertRaises(ValueError):
                    # We need a `value` to set the weight back on CPU
                    set_module_tensor_to_device(model, "batchnorm.running_mean", device1)

                set_module_tensor_to_device(model, "batchnorm.running_mean", device1, value=torch.randn(4))
            else:
                set_module_tensor_to_device(model, "batchnorm.running_mean", device1)
            self.assertEqual(model.batchnorm.running_mean.device, torch.device(device1))

    def test_set_module_tensor_to_meta_and_cpu(self):
        model = ModelForTest()
        self.check_set_module_tensor_for_device(model, "cpu", "meta")

    @require_cuda
    def test_set_module_tensor_to_cpu_and_gpu(self):
        model = ModelForTest()
        self.check_set_module_tensor_for_device(model, "cpu", 0)

    @require_cuda
    def test_set_module_tensor_to_meta_and_gpu(self):
        model = ModelForTest().to(0)
        self.check_set_module_tensor_for_device(model, 0, "meta")

    @require_multi_gpu
    def test_set_module_tensor_between_gpus(self):
        model = ModelForTest().to(0)
        self.check_set_module_tensor_for_device(model, 0, 1)

    def test_set_module_tensor_sets_dtype(self):
        model = ModelForTest()
        set_module_tensor_to_device(model, "linear1.weight", "cpu", value=model.linear1.weight, dtype=torch.float16)
        self.assertEqual(model.linear1.weight.dtype, torch.float16)

    def test_set_module_tensor_checks_shape(self):
        model = ModelForTest()
        tensor = torch.zeros((2, 2))
        with self.assertRaises(ValueError) as cm:
            set_module_tensor_to_device(model, "linear1.weight", "cpu", value=tensor)
        self.assertEqual(
            str(cm.exception),
            'Trying to set a tensor of shape torch.Size([2, 2]) in "weight" (which has shape torch.Size([4, 3])), this look incorrect.',
        )

    def test_named_tensors(self):
        model = nn.BatchNorm1d(4)
        named_tensors = named_module_tensors(model)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"],
        )

        named_tensors = named_module_tensors(model, include_buffers=False)
        self.assertListEqual([name for name, _ in named_tensors], ["weight", "bias"])

        model = ModelForTest()
        named_tensors = named_module_tensors(model)
        self.assertListEqual([name for name, _ in named_tensors], [])

        named_tensors = named_module_tensors(model, recurse=True)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            [
                "linear1.weight",
                "linear1.bias",
                "batchnorm.weight",
                "batchnorm.bias",
                "linear2.weight",
                "linear2.bias",
                "batchnorm.running_mean",
                "batchnorm.running_var",
                "batchnorm.num_batches_tracked",
            ],
        )

        named_tensors = named_module_tensors(model, include_buffers=False, recurse=True)
        self.assertListEqual(
            [name for name, _ in named_tensors],
            ["linear1.weight", "linear1.bias", "batchnorm.weight", "batchnorm.bias", "linear2.weight", "linear2.bias"],
        )

    def test_find_tied_parameters(self):
        model = sequential_model(4)
        self.assertListEqual(find_tied_parameters(model), [])

        model.linear2.weight = model.linear1.weight
        self.assertListEqual(find_tied_parameters(model), [["linear1.weight", "linear2.weight"]])

        model.linear4.weight = model.linear1.weight
        self.assertListEqual(find_tied_parameters(model), [["linear1.weight", "linear2.weight", "linear4.weight"]])

        model = sequential_model(5)
        model.linear1.weight = model.linear4.weight
        model.linear2.weight = model.linear3.weight
        model.linear5.weight = model.linear2.weight
        tied_params = sorted(find_tied_parameters(model), key=lambda x: len(x))
        self.assertListEqual(
            tied_params, [["linear1.weight", "linear4.weight"], ["linear2.weight", "linear3.weight", "linear5.weight"]]
        )

        model = nn.Sequential(OrderedDict([("block1", sequential_model(4)), ("block2", sequential_model(4))]))
        model.block1.linear1.weight = model.block2.linear1.weight
        self.assertListEqual(find_tied_parameters(model), [["block1.linear1.weight", "block2.linear1.weight"]])

    def test_retie_parameters(self):
        model = sequential_model(2)
        retie_parameters(model, [["linear1.weight", "linear2.weight"]])
        self.assertIs(model.linear1.weight, model.linear2.weight)

        model = sequential_model(3)
        retie_parameters(model, [["linear1.weight", "linear2.weight", "linear3.weight"]])

        self.assertIs(model.linear1.weight, model.linear2.weight)
        self.assertIs(model.linear1.weight, model.linear3.weight)

        model = sequential_model(5)
        retie_parameters(
            model, [["linear1.weight", "linear4.weight"], ["linear2.weight", "linear3.weight", "linear5.weight"]]
        )

        self.assertIs(model.linear1.weight, model.linear4.weight)
        self.assertIs(model.linear2.weight, model.linear3.weight)
        self.assertIs(model.linear2.weight, model.linear5.weight)

        model = nn.Sequential(OrderedDict([("block1", sequential_model(4)), ("block2", sequential_model(4))]))
        retie_parameters(model, [["block1.linear1.weight", "block2.linear1.weight"]])

        self.assertIs(model.block1.linear1.weight, model.block2.linear1.weight)

    def test_compute_module_sizes(self):
        model = ModelForTest()
        expected_sizes = {"": 236, "linear1": 64, "linear1.weight": 48, "linear1.bias": 16}
        expected_sizes.update({"linear2": 100, "linear2.weight": 80, "linear2.bias": 20})
        expected_sizes.update({"batchnorm": 72, "batchnorm.weight": 16, "batchnorm.bias": 16})
        expected_sizes.update(
            {"batchnorm.running_mean": 16, "batchnorm.running_var": 16, "batchnorm.num_batches_tracked": 8}
        )

        module_sizes = compute_module_sizes(model)
        self.assertDictEqual(module_sizes, expected_sizes)

        model.half()
        expected_sizes = {k: s // 2 for k, s in expected_sizes.items()}
        # This one is not converted to half.
        expected_sizes["batchnorm.num_batches_tracked"] = 8
        # This impacts batchnorm and total
        expected_sizes["batchnorm"] += 4
        expected_sizes[""] += 4

        module_sizes = compute_module_sizes(model)
        self.assertDictEqual(module_sizes, expected_sizes)

    def test_check_device_map(self):
        model = ModelForTest()
        check_device_map(model, {"": 0})
        with self.assertRaises(ValueError):
            check_device_map(model, {"linear1": 0, "linear2": 1})

        check_device_map(model, {"linear1": 0, "linear2": 1, "batchnorm": 1})

    def shard_test_model(self, model, tmp_dir):
        module_index = {
            "linear1": "checkpoint_part1.bin",
            "batchnorm": "checkpoint_part2.bin",
            "linear2": "checkpoint_part3.bin",
        }
        index = {}
        for name, _ in model.state_dict().items():
            module = name.split(".")[0]
            index[name] = module_index[module]

        with open(os.path.join(tmp_dir, "weight_map.index.json"), "w") as f:
            json.dump(index, f)

        for module, fname in module_index.items():
            state_dict = {k: v for k, v in model.state_dict().items() if k.startswith(module)}
            full_fname = os.path.join(tmp_dir, fname)
            torch.save(state_dict, full_fname)

    def test_load_checkpoint_in_model(self):
        # Check with whole checkpoint
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), fname)
            load_checkpoint_in_model(model, fname)

        # Check with sharded index
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            index_file = os.path.join(tmp_dir, "weight_map.index.json")
            load_checkpoint_in_model(model, index_file)

        # Check with sharded checkpoint
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            load_checkpoint_in_model(model, tmp_dir)

    @require_cuda
    def test_load_checkpoint_in_model_one_gpu(self):
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": "cpu"}

        # Check with whole checkpoint
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), fname)
            load_checkpoint_in_model(model, fname, device_map=device_map)
        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # Check with sharded index
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            index_file = os.path.join(tmp_dir, "weight_map.index.json")
            load_checkpoint_in_model(model, index_file, device_map=device_map)

        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # Check with sharded checkpoint folder
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            load_checkpoint_in_model(model, tmp_dir, device_map=device_map)

        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

    @require_cuda
    def test_load_checkpoint_in_model_disk_offload(self):
        device_map = {"linear1": "cpu", "batchnorm": "disk", "linear2": "cpu"}

        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), fname)
            load_checkpoint_in_model(model, fname, device_map=device_map, offload_folder=tmp_dir)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        # Buffers are not offloaded by default
        self.assertEqual(model.batchnorm.running_mean.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), fname)
            load_checkpoint_in_model(model, fname, device_map=device_map, offload_folder=tmp_dir, offload_buffers=True)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.running_mean.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

    @require_multi_gpu
    def test_load_checkpoint_in_model_two_gpu(self):
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": 1}

        # Check with whole checkpoint
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), fname)
            load_checkpoint_in_model(model, fname, device_map=device_map)
        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device(1))

        # Check with sharded index
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            index_file = os.path.join(tmp_dir, "weight_map.index.json")
            load_checkpoint_in_model(model, index_file, device_map=device_map)

        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device(1))

        # Check with sharded checkpoint
        model = ModelForTest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.shard_test_model(model, tmp_dir)
            load_checkpoint_in_model(model, tmp_dir, device_map=device_map)

        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device(1))

    def test_clean_device_map(self):
        # Regroup everything if all is on the same device
        self.assertDictEqual(clean_device_map({"a": 0, "b": 0, "c": 0}), {"": 0})
        # Regroups children of level 1 on the same device
        self.assertDictEqual(
            clean_device_map({"a.x": 0, "a.y": 0, "b.x": 1, "b.y": 1, "c": 1}), {"a": 0, "b": 1, "c": 1}
        )
        # Regroups children of level 2 on the same device
        self.assertDictEqual(
            clean_device_map({"a.x": 0, "a.y": 0, "b.x.0": 1, "b.x.1": 1, "b.y.0": 2, "b.y.1": 2, "c": 2}),
            {"a": 0, "b.x": 1, "b.y": 2, "c": 2},
        )

    def test_infer_auto_device_map(self):
        model = ModelForTest()
        # model has size 236: linear1 64, batchnorm 72, linear2 100

        device_map = infer_auto_device_map(model, max_memory={0: 200, 1: 200})
        # only linear1 fits on device 0 as we keep memory available for the maximum layer in case of offload
        self.assertDictEqual(device_map, {"linear1": 0, "batchnorm": 1, "linear2": 1})

        device_map = infer_auto_device_map(model, max_memory={0: 200, 1: 172, 2: 200})
        # On device 1, we don't care about keeping size available for the max layer, so even if there is just the
        # size available for batchnorm + linear2, they fit here.
        self.assertDictEqual(device_map, {"linear1": 0, "batchnorm": 1, "linear2": 1})

        model.linear1.weight = model.linear2.weight
        device_map = infer_auto_device_map(model, max_memory={0: 200, 1: 200})
        # By tying weights, the whole model fits on device 0
        self.assertDictEqual(device_map, {"": 0})

        # When splitting a bigger model, the split is done at the layer level
        model = nn.Sequential(ModelForTest(), ModelForTest(), ModelForTest())
        device_map = infer_auto_device_map(model, max_memory={0: 500, 1: 500})
        self.assertDictEqual(device_map, {"0": 0, "1.linear1": 0, "1.batchnorm": 0, "1.linear2": 1, "2": 1})

        # With no_split_module_classes, it's done at that module level
        model = nn.Sequential(ModelForTest(), ModelForTest(), ModelForTest())
        device_map = infer_auto_device_map(
            model, max_memory={0: 500, 1: 500}, no_split_module_classes=["ModelForTest"]
        )
        self.assertDictEqual(device_map, {"0": 0, "1": 1, "2": 1})

    def test_infer_auto_device_map_with_tied_weights(self):
        model = nn.Sequential(
            OrderedDict([("layer1", ModelForTest()), ("layer2", ModelForTest()), ("layer3", ModelForTest())])
        )
        model.layer3.linear2.weight = model.layer1.linear2.weight
        device_map = infer_auto_device_map(model, max_memory={0: 400, 1: 500})
        expected = {"layer1": 0, "layer3.linear2": 0, "layer2": 1, "layer3.linear1": 1, "layer3.batchnorm": 1}
        self.assertDictEqual(device_map, expected)

        # With three weights tied together
        model.layer2.linear2.weight = model.layer1.linear2.weight
        device_map = infer_auto_device_map(model, max_memory={0: 400, 1: 500})
        expected = {
            "layer1": 0,
            "layer2.linear2": 0,
            "layer3.linear2": 0,
            "layer2.linear1": 1,
            "layer2.batchnorm": 1,
            "layer3.linear1": 1,
            "layer3.batchnorm": 1,
        }
        self.assertDictEqual(device_map, expected)

        # With two groups of weights tied together
        model.layer2.linear1.weight = model.layer1.linear1.weight
        device_map = infer_auto_device_map(model, max_memory={0: 400, 1: 500})
        expected = {
            "layer1": 0,
            "layer2.linear1": 0,
            "layer2.linear2": 0,
            "layer3.linear2": 0,
            "layer2.batchnorm": 1,
            "layer3.linear1": 1,
            "layer3.batchnorm": 1,
        }
        self.assertDictEqual(device_map, expected)

        # With weights ties in the same module
        model = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(4, 4)),
                    ("linear2", nn.Linear(6, 6)),
                    ("linear3", nn.Linear(4, 4)),
                    ("linear4", nn.Linear(6, 6)),
                ]
            )
        )
        model.linear3.weight = model.linear1.weight
        model.linear3.bias = model.linear1.bias
        device_map = infer_auto_device_map(model, max_memory={0: 250, 1: 400})
        expected = {"linear1": 0, "linear2": 1, "linear3": 0, "linear4": 1}
        self.assertDictEqual(device_map, expected)

    @require_huggingface_suite
    def test_infer_auto_device_map_on_t0pp(self):
        from transformers import AutoConfig, AutoModelForSeq2SeqLM

        config = AutoConfig.from_pretrained("bigscience/T0pp")
        with init_empty_weights():
            model = AutoModelForSeq2SeqLM.from_config(config)
        model.tie_weights()

        special_dtypes = {n: torch.float32 for n, _ in model.named_parameters() if "wo" in n}
        max_memory = {0: 10**10, 1: 10**10, "cpu": 10**10}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=["T5Block"],
            dtype=torch.float16,
            max_memory=max_memory,
            special_dtypes=special_dtypes,
        )

        # The 3 tied weights should all be on device 0
        self.assertEqual(device_map["shared"], 0)
        self.assertEqual(device_map["encoder.embed_tokens"], 0)
        self.assertEqual(device_map["decoder.embed_tokens"], 0)

    @require_cuda
    def test_get_balanced_memory(self):
        model = ModelForTest()
        # model has size 236: linear1 64, batchnorm 72, linear2 100
        max_memory = get_balanced_memory(model, max_memory={0: 200, 1: 200})
        self.assertDictEqual({0: 200, 1: 200}, max_memory)

        # We should be able to set models on a non-contiguous sub-set of
        max_memory = get_balanced_memory(model, max_memory={0: 200, 2: 200})
        self.assertDictEqual({0: 200, 2: 200}, max_memory)

        max_memory = get_balanced_memory(model, max_memory={0: 300, 1: 300})
        self.assertDictEqual({0: 215, 1: 300}, max_memory)

        # Last device always get max memory to give more buffer and avoid accidental CPU offload
        max_memory = get_balanced_memory(model, max_memory={0: 300, 1: 500})
        self.assertDictEqual({0: 215, 1: 500}, max_memory)

        # Last device always get max memory to give more buffer, even if CPU is provided
        max_memory = get_balanced_memory(model, max_memory={0: 300, "cpu": 1000})
        self.assertDictEqual({0: 300, "cpu": 1000}, max_memory)

        # If we set a device to 0, it's not counted.
        max_memory = get_balanced_memory(model, max_memory={0: 0, 1: 300, 2: 300})
        self.assertDictEqual({0: 0, 1: 215, 2: 300}, max_memory)

    @require_cuda
    @require_safetensors
    def test_load_state_dict(self):
        from safetensors.torch import save_file

        state_dict = {k: torch.randn(4, 5) for k in ["a", "b", "c"]}
        device_maps = [{"a": "cpu", "b": 0, "c": "disk"}, {"a": 0, "b": 0, "c": "disk"}, {"a": 0, "b": 0, "c": 0}]

        for device_map in device_maps:
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_file = os.path.join(tmp_dir, "model.safetensors")
                save_file(state_dict, checkpoint_file, metadata={"format": "pt"})

                loaded_state_dict = load_state_dict(checkpoint_file, device_map=device_map)

            for param, device in device_map.items():
                device = device if device != "disk" else "cpu"
                self.assertEqual(loaded_state_dict[param].device, torch.device(device))

    def test_convert_file_size(self):
        result = convert_file_size_to_int("100MB")
        self.assertEqual(result, 100 * (10**6))

        result = convert_file_size_to_int("2GiB")
        self.assertEqual(result, 2 * (2**30))

        result = convert_file_size_to_int("512KiB")
        self.assertEqual(result, 512 * (2**10))

        result = convert_file_size_to_int("1.5GB")
        self.assertEqual(result, 1.5 * (10**9))

        result = convert_file_size_to_int("100KB")
        self.assertEqual(result, 100 * (10**3))

        result = convert_file_size_to_int(500)
        self.assertEqual(result, 500)

        with self.assertRaises(ValueError):
            convert_file_size_to_int("5MBB")

        with self.assertRaises(ValueError):
            convert_file_size_to_int("5k0MB")

        with self.assertRaises(ValueError):
            convert_file_size_to_int("-1GB")
