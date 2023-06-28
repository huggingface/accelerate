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
import unittest

import torch
import torch.nn as nn

from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    attach_align_device_hook,
    remove_hook_from_module,
    remove_hook_from_submodules,
)
from accelerate.test_utils import require_multi_gpu


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class PreForwardHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
        return (args[0] + 1,) + args[1:], kwargs


class PostForwardHook(ModelHook):
    def post_forward(self, module, output):
        return output + 1


class HooksModelTester(unittest.TestCase):
    def test_add_and_remove_hooks(self):
        test_model = ModelForTest()
        test_hook = ModelHook()

        add_hook_to_module(test_model, test_hook)
        self.assertEqual(test_model._hf_hook, test_hook)
        self.assertTrue(hasattr(test_model, "_old_forward"))

        # Check adding the hook did not change the name or the signature
        self.assertEqual(test_model.forward.__name__, "forward")
        self.assertListEqual(list(inspect.signature(test_model.forward).parameters), ["x"])

        remove_hook_from_module(test_model)
        self.assertFalse(hasattr(test_model, "_hf_hook"))
        self.assertFalse(hasattr(test_model, "_old_forward"))

    def test_append_and_remove_hooks(self):
        test_model = ModelForTest()
        test_hook = ModelHook()

        add_hook_to_module(test_model, test_hook)
        add_hook_to_module(test_model, test_hook, append=True)

        self.assertEqual(isinstance(test_model._hf_hook, SequentialHook), True)
        self.assertEqual(len(test_model._hf_hook.hooks), 2)
        self.assertTrue(hasattr(test_model, "_old_forward"))

        # Check adding the hook did not change the name or the signature
        self.assertEqual(test_model.forward.__name__, "forward")
        self.assertListEqual(list(inspect.signature(test_model.forward).parameters), ["x"])

        remove_hook_from_module(test_model)
        self.assertFalse(hasattr(test_model, "_hf_hook"))
        self.assertFalse(hasattr(test_model, "_old_forward"))

    def test_pre_forward_hook_is_executed(self):
        test_model = ModelForTest()
        x = torch.randn(2, 3)
        expected = test_model(x + 1)
        expected2 = test_model(x + 2)

        test_hook = PreForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        self.assertTrue(torch.allclose(output1, expected, atol=1e-5))

        # Attaching a hook to a model when it already has one replaces, does not chain
        test_hook = PreForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        self.assertTrue(torch.allclose(output1, expected, atol=1e-5))

        # You need to use the sequential hook to chain two or more hooks
        test_hook = SequentialHook(PreForwardHook(), PreForwardHook())
        add_hook_to_module(test_model, test_hook)

        output2 = test_model(x)
        assert torch.allclose(output2, expected2, atol=1e-5)

    def test_post_forward_hook_is_executed(self):
        test_model = ModelForTest()
        x = torch.randn(2, 3)
        output = test_model(x)

        test_hook = PostForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        self.assertTrue(torch.allclose(output1, output + 1, atol=1e-5))

        # Attaching a hook to a model when it already has one replaces, does not chain
        test_hook = PostForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        self.assertTrue(torch.allclose(output1, output + 1, atol=1e-5))

        # You need to use the sequential hook to chain two or more hooks
        test_hook = SequentialHook(PostForwardHook(), PostForwardHook())
        add_hook_to_module(test_model, test_hook)

        output2 = test_model(x)
        assert torch.allclose(output2, output + 2, atol=1e-5)

    def test_no_grad_in_hook(self):
        test_model = ModelForTest()
        x = torch.randn(2, 3)
        output = test_model(x)

        test_hook = PostForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        self.assertTrue(torch.allclose(output1, output + 1))
        self.assertTrue(output1.requires_grad)

        test_hook.no_grad = True
        output1 = test_model(x)
        self.assertFalse(output1.requires_grad)

    @require_multi_gpu
    def test_align_devices_as_model_parallelism(self):
        model = ModelForTest()
        # Everything is on CPU
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # This will move each submodule on different devices
        add_hook_to_module(model.linear1, AlignDevicesHook(execution_device=0))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(execution_device=0))
        add_hook_to_module(model.linear2, AlignDevicesHook(execution_device=1))

        self.assertEqual(model.linear1.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.weight.device, torch.device(0))
        self.assertEqual(model.batchnorm.running_mean.device, torch.device(0))
        self.assertEqual(model.linear2.weight.device, torch.device(1))

        # We can still make a forward pass. The input does not need to be on any particular device
        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, torch.device(1))

        # We can add a general hook to put back output on same device as input.
        add_hook_to_module(model, AlignDevicesHook(io_same_device=True))
        x = torch.randn(2, 3).to(0)
        output = model(x)
        self.assertEqual(output.device, torch.device(0))

    def test_align_devices_as_cpu_offload(self):
        model = ModelForTest()

        # Everything is on CPU
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # This will move each submodule on different devices
        hook_kwargs = {"execution_device": 0 if torch.cuda.is_available() else "cpu", "offload": True}

        add_hook_to_module(model.linear1, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.linear2, AlignDevicesHook(**hook_kwargs))

        # Parameters have been offloaded, so on the meta device
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(hook_kwargs["execution_device"])
        self.assertEqual(model.batchnorm.running_mean.device, device)

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_module(model.linear1)
        remove_hook_from_module(model.batchnorm)
        remove_hook_from_module(model.linear2)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # Now test with buffers included in the offload
        hook_kwargs = {
            "execution_device": 0 if torch.cuda.is_available() else "cpu",
            "offload": True,
            "offload_buffers": True,
        }

        add_hook_to_module(model.linear1, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.linear2, AlignDevicesHook(**hook_kwargs))

        # Parameters have been offloaded, so on the meta device, buffers included
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.running_mean.device, torch.device("meta"))

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_module(model.linear1)
        remove_hook_from_module(model.batchnorm)
        remove_hook_from_module(model.linear2)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

    def test_attach_align_device_hook_as_cpu_offload(self):
        model = ModelForTest()

        # Everything is on CPU
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # This will move each submodule on different devices
        execution_device = 0 if torch.cuda.is_available() else "cpu"
        attach_align_device_hook(model, execution_device=execution_device, offload=True)

        # Parameters have been offloaded, so on the meta device
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(execution_device)
        self.assertEqual(model.batchnorm.running_mean.device, device)

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # Now test with buffers included in the offload
        attach_align_device_hook(model, execution_device=execution_device, offload=True, offload_buffers=True)

        # Parameters have been offloaded, so on the meta device, buffers included
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.running_mean.device, torch.device("meta"))

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

    def test_attach_align_device_hook_as_cpu_offload_with_weight_map(self):
        model = ModelForTest()

        # Everything is on CPU
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # This will move each submodule on different devices
        execution_device = 0 if torch.cuda.is_available() else "cpu"
        attach_align_device_hook(
            model, execution_device=execution_device, offload=True, weights_map=model.state_dict()
        )

        # Parameters have been offloaded, so on the meta device
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(execution_device)
        self.assertEqual(model.batchnorm.running_mean.device, device)

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))

        # Now test with buffers included in the offload
        attach_align_device_hook(
            model,
            execution_device=execution_device,
            offload=True,
            weights_map=model.state_dict(),
            offload_buffers=True,
        )

        # Parameters have been offloaded, so on the meta device, buffers included
        self.assertEqual(model.linear1.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("meta"))
        self.assertEqual(model.linear2.weight.device, torch.device("meta"))
        self.assertEqual(model.batchnorm.running_mean.device, torch.device("meta"))

        x = torch.randn(2, 3)
        output = model(x)
        self.assertEqual(output.device, device)

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        self.assertEqual(model.linear1.weight.device, torch.device("cpu"))
        self.assertEqual(model.batchnorm.weight.device, torch.device("cpu"))
        self.assertEqual(model.linear2.weight.device, torch.device("cpu"))
