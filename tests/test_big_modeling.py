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

import unittest
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from accelerate.big_modeling import cpu_offload, disk_offload, dispatch_model, init_empty_weights
from accelerate.hooks import remove_hook_from_submodules
from accelerate.test_utils import require_cuda, require_multi_gpu, slow
from accelerate.utils import offload_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class BiggerModelForTest(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = nn.Linear(5, 6)
        self.linear4 = nn.Linear(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class BigModelingTester(unittest.TestCase):
    def test_init_empty_weights(self):
        # base use
        with init_empty_weights():
            module = nn.Linear(4, 5)
        self.assertEqual(module.weight.device, torch.device("meta"))

        # base use with buffers, they are not touched
        with init_empty_weights():
            module = nn.BatchNorm1d(4)
        self.assertEqual(module.weight.device, torch.device("meta"))
        self.assertEqual(module.running_mean.device, torch.device("cpu"))

        # Use with include_buffers=True
        with init_empty_weights(include_buffers=True):
            module = nn.BatchNorm1d(4)
        self.assertEqual(module.weight.device, torch.device("meta"))
        self.assertEqual(module.running_mean.device, torch.device("meta"))

        # Double check we didn't break PyTorch
        module = nn.BatchNorm1d(4)
        self.assertEqual(module.weight.device, torch.device("cpu"))
        self.assertEqual(module.running_mean.device, torch.device("cpu"))

    def test_init_empty_weights_very_large_model(self):
        # This is a 100 billion parameters model.
        with init_empty_weights():
            _ = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])

    def test_cpu_offload(self):
        model = ModelForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        cpu_offload(model, execution_device=device)
        output = model(x)
        self.assertTrue(torch.allclose(expected, output.cpu()))

        # Clean up for next test.
        remove_hook_from_submodules(model)

        cpu_offload(model, execution_device=device, offload_buffers=True)
        output = model(x)
        self.assertTrue(torch.allclose(expected, output.cpu()))

    @slow
    @require_cuda
    def test_cpu_offload_gpt2(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        cpu_offload(gpt2, execution_device=0)
        outputs = gpt2.generate(inputs["input_ids"])
        self.assertEqual(
            tokenizer.decode(outputs[0].tolist()),
            "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo",
        )

    def test_disk_offload(self):
        model = ModelForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        with TemporaryDirectory() as tmp_dir:
            disk_offload(model, tmp_dir, execution_device=device)
            output = model(x)
            self.assertTrue(torch.allclose(expected, output.cpu()))

            # Clean up for next test.
            remove_hook_from_submodules(model)

        with TemporaryDirectory() as tmp_dir:
            disk_offload(model, tmp_dir, execution_device=device, offload_buffers=True)
            output = model(x)
            self.assertTrue(torch.allclose(expected, output.cpu()))

    @slow
    @require_cuda
    def test_disk_offload_gpt2(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        with TemporaryDirectory() as tmp_dir:
            disk_offload(gpt2, tmp_dir, execution_device=0)
            outputs = gpt2.generate(inputs["input_ids"])
            self.assertEqual(
                tokenizer.decode(outputs[0].tolist()),
                "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo",
            )

    @require_cuda
    def test_dispatch_model(self):
        model = ModelForTest()
        device_map = {"linear1": "disk", "batchnorm": "cpu", "linear2": 0}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            output = model(x)
            self.assertTrue(torch.allclose(expected, output.cpu(), atol=1e-5))

    @require_multi_gpu
    def test_dispatch_model_multi_gpu(self):
        model = BiggerModelForTest()
        device_map = {"linear1": "cpu", "linear2": "disk", "batchnorm": "cpu", "linear3": 0, "linear4": 1}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            output = model(x)
            self.assertTrue(torch.allclose(expected, output.cpu(), atol=1e-5))

    @slow
    @require_multi_gpu
    def test_dispatch_model_gpt2_on_two_gpus(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        # Dispatch on GPUs 0 and 1
        device_map = {
            "transformer.wte": 0,
            "transformer.wpe": 0,
            "transformer.ln_f": 1,
            "lm_head": 1,
        }
        for i in range(12):
            device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

        gpt2 = dispatch_model(gpt2, device_map)
        outputs = gpt2.generate(inputs["input_ids"])
        self.assertEqual(
            tokenizer.decode(outputs[0].tolist()),
            "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo",
        )

        # Dispatch with a bit of CPU offload
        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        for i in range(4):
            device_map[f"transformer.h.{i}"] = "cpu"
        gpt2 = dispatch_model(gpt2, device_map)
        outputs = gpt2.generate(inputs["input_ids"])
        self.assertEqual(
            tokenizer.decode(outputs[0].tolist()),
            "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo",
        )
        # Dispatch with a bit of CPU and disk offload
        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        for i in range(2):
            device_map[f"transformer.h.{i}"] = "disk"

        with TemporaryDirectory() as tmp_dir:
            state_dict = {
                k: p for k, p in gpt2.state_dict().items() if "transformer.h.0" in k or "transformer.h.1" in k
            }
            offload_state_dict(tmp_dir, state_dict)
            gpt2 = dispatch_model(gpt2, device_map, offload_dir=tmp_dir)
            outputs = gpt2.generate(inputs["input_ids"])
            self.assertEqual(
                tokenizer.decode(outputs[0].tolist()),
                "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo",
            )
