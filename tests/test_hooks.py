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
import re
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.fx import symbolic_trace

from accelerate.big_modeling import attach_layerwise_casting_hooks
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    attach_align_device_hook,
    remove_hook_from_module,
    remove_hook_from_submodules,
)
from accelerate.test_utils import require_multi_device, require_non_hpu, torch_device
from accelerate.utils.constants import SUPPORTED_PYTORCH_LAYERS_FOR_UPCASTING


torch_device = f"{torch_device}:0" if torch_device != "cpu" else "cpu"


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
    def check_dtype_for_layerwise_upcasting(
        self,
        module,
        storage_dtype,
        loading_type,
        patterns_to_check=None,
    ):
        for name, submodule in module.named_modules():
            attrs = []
            if getattr(submodule, "weight", None) is not None:
                attrs.append(("weight", submodule.weight))
            if getattr(submodule, "bias", None) is not None:
                attrs.append(("bias", submodule.bias))

            if not isinstance(submodule, SUPPORTED_PYTORCH_LAYERS_FOR_UPCASTING):
                if patterns_to_check is None:
                    for _, tensor in attrs:
                        self.assertEqual(tensor.dtype, loading_type)
                continue

            if patterns_to_check and any(re.search(pat, name) for pat in patterns_to_check):
                expected = loading_type
            else:
                expected = storage_dtype

            for _, tensor in attrs:
                self.assertEqual(tensor.dtype, expected)

    def test_add_and_remove_hooks(self):
        test_model = ModelForTest()
        test_hook = ModelHook()

        add_hook_to_module(test_model, test_hook)
        assert test_model._hf_hook == test_hook
        assert hasattr(test_model, "_old_forward")

        # Check adding the hook did not change the name or the signature
        assert test_model.forward.__name__ == "forward"
        assert list(inspect.signature(test_model.forward).parameters) == ["x"]

        remove_hook_from_module(test_model)
        assert not hasattr(test_model, "_hf_hook")
        assert not hasattr(test_model, "_old_forward")

    def test_append_and_remove_hooks(self):
        test_model = ModelForTest()
        test_hook = ModelHook()

        add_hook_to_module(test_model, test_hook)
        add_hook_to_module(test_model, test_hook, append=True)

        assert isinstance(test_model._hf_hook, SequentialHook) is True
        assert len(test_model._hf_hook.hooks) == 2
        assert hasattr(test_model, "_old_forward")

        # Check adding the hook did not change the name or the signature
        assert test_model.forward.__name__ == "forward"
        assert list(inspect.signature(test_model.forward).parameters) == ["x"]

        remove_hook_from_module(test_model)
        assert not hasattr(test_model, "_hf_hook")
        assert not hasattr(test_model, "_old_forward")

    def test_pre_forward_hook_is_executed(self):
        test_model = ModelForTest()
        x = torch.randn(2, 3)
        expected = test_model(x + 1)
        expected2 = test_model(x + 2)

        test_hook = PreForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        assert torch.allclose(output1, expected, atol=1e-5)

        # Attaching a hook to a model when it already has one replaces, does not chain
        test_hook = PreForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        assert torch.allclose(output1, expected, atol=1e-5)

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
        assert torch.allclose(output1, (output + 1), atol=1e-5)

        # Attaching a hook to a model when it already has one replaces, does not chain
        test_hook = PostForwardHook()
        add_hook_to_module(test_model, test_hook)
        output1 = test_model(x)
        assert torch.allclose(output1, (output + 1), atol=1e-5)

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
        assert torch.allclose(output1, (output + 1))
        assert output1.requires_grad

        test_hook.no_grad = True
        output1 = test_model(x)
        assert not output1.requires_grad

    @require_non_hpu  # hpu does not support device indexing "hpu:1"
    @require_multi_device
    def test_align_devices_as_model_parallelism(self):
        model = ModelForTest()
        # Everything is on CPU
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # This will move each submodule on different devices
        add_hook_to_module(model.linear1, AlignDevicesHook(execution_device=0))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(execution_device=0))
        add_hook_to_module(model.linear2, AlignDevicesHook(execution_device=1))

        assert model.linear1.weight.device == torch.device(torch_device)
        assert model.batchnorm.weight.device == torch.device(torch_device)
        assert model.batchnorm.running_mean.device == torch.device(torch_device)
        assert model.linear2.weight.device == torch.device(torch_device.replace(":0", ":1"))

        # We can still make a forward pass. The input does not need to be on any particular device
        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == torch.device(torch_device.replace(":0", ":1"))

        # We can add a general hook to put back output on same device as input.
        add_hook_to_module(model, AlignDevicesHook(io_same_device=True))
        x = torch.randn(2, 3).to(torch_device)
        output = model(x)
        assert output.device == torch.device(torch_device)

    def test_align_devices_as_cpu_offload(self):
        model = ModelForTest()

        # Everything is on CPU
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # This will move each submodule on different devices
        hook_kwargs = {"execution_device": torch_device, "offload": True}

        add_hook_to_module(model.linear1, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.linear2, AlignDevicesHook(**hook_kwargs))

        # Parameters have been offloaded, so on the meta device
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(hook_kwargs["execution_device"])
        assert model.batchnorm.running_mean.device == device

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_module(model.linear1)
        remove_hook_from_module(model.batchnorm)
        remove_hook_from_module(model.linear2)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # Now test with buffers included in the offload
        hook_kwargs = {
            "execution_device": torch_device,
            "offload": True,
            "offload_buffers": True,
        }

        add_hook_to_module(model.linear1, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.batchnorm, AlignDevicesHook(**hook_kwargs))
        add_hook_to_module(model.linear2, AlignDevicesHook(**hook_kwargs))

        # Parameters have been offloaded, so on the meta device, buffers included
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        assert model.batchnorm.running_mean.device == torch.device("meta")

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_module(model.linear1)
        remove_hook_from_module(model.batchnorm)
        remove_hook_from_module(model.linear2)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

    def test_attach_align_device_hook_as_cpu_offload(self):
        model = ModelForTest()

        # Everything is on CPU
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # This will move each submodule on different devices
        execution_device = torch_device
        attach_align_device_hook(model, execution_device=execution_device, offload=True)

        # Parameters have been offloaded, so on the meta device
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(execution_device)
        assert model.batchnorm.running_mean.device == device

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # Now test with buffers included in the offload
        attach_align_device_hook(model, execution_device=execution_device, offload=True, offload_buffers=True)

        # Parameters have been offloaded, so on the meta device, buffers included
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        assert model.batchnorm.running_mean.device == torch.device("meta")

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

    def test_attach_align_device_hook_as_cpu_offload_with_weight_map(self):
        model = ModelForTest()

        # Everything is on CPU
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # This will move each submodule on different devices
        execution_device = torch_device
        attach_align_device_hook(
            model, execution_device=execution_device, offload=True, weights_map=model.state_dict()
        )

        # Parameters have been offloaded, so on the meta device
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        # Buffers are not included in the offload by default, so are on the execution device
        device = torch.device(execution_device)
        assert model.batchnorm.running_mean.device == device

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

        # Now test with buffers included in the offload
        attach_align_device_hook(
            model,
            execution_device=execution_device,
            offload=True,
            weights_map=model.state_dict(),
            offload_buffers=True,
        )

        # Parameters have been offloaded, so on the meta device, buffers included
        assert model.linear1.weight.device == torch.device("meta")
        assert model.batchnorm.weight.device == torch.device("meta")
        assert model.linear2.weight.device == torch.device("meta")
        assert model.batchnorm.running_mean.device == torch.device("meta")

        x = torch.randn(2, 3)
        output = model(x)
        assert output.device == device

        # Removing hooks loads back the weights in the model.
        remove_hook_from_submodules(model)
        assert model.linear1.weight.device == torch.device("cpu")
        assert model.batchnorm.weight.device == torch.device("cpu")
        assert model.linear2.weight.device == torch.device("cpu")

    def test_add_remove_hook_fx_graph_module(self):
        with torch.no_grad():
            test_model = ModelForTest()
            test_hook = ModelHook()

            x = torch.randn(2, 3)
            output1 = test_model(x)

            graph_model = symbolic_trace(test_model)

            output2 = graph_model(x)

            assert torch.allclose(output1, output2)

            add_hook_to_module(graph_model, test_hook)
            remove_hook_from_module(graph_model, recurse=True)

            # We want to make sure that `add_hook_to_module` and `remove_hook_from_module` yields back an fx.GraphModule
            # that behaves correctly (for example that is not frozen, see https://github.com/huggingface/accelerate/pull/2369).
            # For that, we add a sigmoid node to the FX graph and make sure that the new output (output3 below) is different than
            # the original model's output.
            linear2_node = None
            for node in graph_model.graph.nodes:
                if node.name == "linear2":
                    linear2_node = node
            assert linear2_node is not None

            graph_model.graph.inserting_after(linear2_node)
            new_node = graph_model.graph.create_node(
                op="call_function", target=torch.sigmoid, args=(linear2_node,), name="relu"
            )

            output_node = None
            for node in graph_model.graph.nodes:
                if node.name == "output":
                    output_node = node
            assert output_node is not None

            output_node.replace_input_with(linear2_node, new_node)

            graph_model.graph.lint()
            graph_model.recompile()

            output3 = graph_model(x)

            # Now the output is expected to be different since we modified the graph.
            assert not torch.allclose(output1, output3)

    @parameterized.expand(
        [
            (torch.float16, torch.float32),
            (torch.float8_e4m3fn, torch.float32),
            (torch.float8_e4m3fn, torch.float32, ["batchnorm"]),
        ]
    )
    def test_layerwise_upcasting_inference(self, storage_dtype, compute_dtype, skip_modules_pattern=None):
        test_model = ModelForTest()
        loading_dtype = next(test_model.parameters()).data.dtype
        inputs = torch.randn(2, 3)
        inputs = inputs.to(compute_dtype) if inputs.dtype == torch.float32 else inputs

        attach_layerwise_casting_hooks(
            test_model,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
            skip_modules_pattern=skip_modules_pattern,
        )
        patterns_to_check = skip_modules_pattern if skip_modules_pattern else None
        self.check_dtype_for_layerwise_upcasting(test_model, storage_dtype, loading_dtype, patterns_to_check)

        with torch.no_grad():
            _ = test_model(inputs)
