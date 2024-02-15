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
import copy
import gc
import os
import unittest
from collections import OrderedDict
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate.big_modeling import (
    cpu_offload,
    cpu_offload_with_hook,
    disk_offload,
    dispatch_model,
    init_empty_weights,
    init_on_device,
    load_checkpoint_and_dispatch,
)
from accelerate.hooks import remove_hook_from_submodules
from accelerate.test_utils import (
    require_bnb,
    require_cuda,
    require_mps,
    require_multi_gpu,
    require_non_torch_xla,
    slow,
)
from accelerate.utils import is_torch_version, offload_state_dict


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class LinearWithNonPersistentBuffers(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.ones((out_features, in_features), **factory_kwargs))
        if bias:
            self.register_buffer("bias", torch.ones(out_features, **factory_kwargs), persistent=False)
        else:
            self.register_buffer("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)


class ModelForTestNonPersistentBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithNonPersistentBuffers(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = LinearWithNonPersistentBuffers(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class ModelForTestCopy(nn.Module):
    def __init__(self, id: int):
        super().__init__()
        self.id = id
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x))), self.id


class ModelForTestTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class BiggerModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = nn.Linear(5, 6)
        self.linear4 = nn.Linear(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


# To test preload_module_classes
class ModuleWithUnusedSubModules(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return x @ self.linear.weight.t() + self.linear.bias


class ModelWithUnusedSubModulesForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = ModuleWithUnusedSubModules(3, 4)
        self.linear2 = ModuleWithUnusedSubModules(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = ModuleWithUnusedSubModules(5, 6)
        self.linear4 = ModuleWithUnusedSubModules(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class BigModelingTester(unittest.TestCase):
    def test_init_empty_weights(self):
        # base use
        with init_empty_weights():
            module = nn.Linear(4, 5)
        assert module.weight.device == torch.device("meta")

        # base use with buffers, they are not touched
        with init_empty_weights():
            module = nn.BatchNorm1d(4)
        assert module.weight.device == torch.device("meta")
        assert module.running_mean.device == torch.device("cpu")

        # Use with include_buffers=True
        register_parameter_func = nn.Module.register_parameter
        register_buffer_func = nn.Module.register_buffer
        with init_empty_weights(include_buffers=True):
            module = nn.BatchNorm1d(4)
            # nn.Module.register_parameter/buffer shouldn't be changed with torch >= 2.0
            if is_torch_version(">=", "2.0"):
                assert register_parameter_func == nn.Module.register_parameter
                assert register_buffer_func == nn.Module.register_buffer
        assert module.weight.device == torch.device("meta")
        assert module.running_mean.device == torch.device("meta")

        # Double check we didn't break PyTorch
        module = nn.BatchNorm1d(4)
        assert module.weight.device == torch.device("cpu")
        assert module.running_mean.device == torch.device("cpu")

    def test_init_empty_weights_very_large_model(self):
        # This is a 100 billion parameters model.
        with init_empty_weights():
            _ = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])

    @require_cuda
    def test_init_on_device_cuda(self):
        device = torch.device("cuda:0")
        with init_on_device(device):
            model = nn.Linear(10, 10)
        assert model.weight.device == device
        assert model.weight.device == device

    @require_mps
    def test_init_on_device_mps(self):
        device = torch.device("mps:0")
        with init_on_device(device):
            model = nn.Linear(10, 10)
        assert model.weight.device == device
        assert model.weight.device == device

    def test_cpu_offload(self):
        model = ModelForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        cpu_offload(model, execution_device=device)
        output = model(x)
        assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

        # Clean up for next test.
        remove_hook_from_submodules(model)

        cpu_offload(model, execution_device=device, offload_buffers=True)
        output = model(x)
        assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

    def test_cpu_offload_with_unused_submodules(self):
        model = ModelWithUnusedSubModulesForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        cpu_offload(model, execution_device=device, preload_module_classes=["ModuleWithUnusedSubModules"])
        output = model(x)
        assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

        # Clean up for next test.
        remove_hook_from_submodules(model)

        cpu_offload(
            model,
            execution_device=device,
            offload_buffers=True,
            preload_module_classes=["ModuleWithUnusedSubModules"],
        )
        output = model(x)
        assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

    @slow
    @require_cuda
    def test_cpu_offload_gpt2(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        cpu_offload(gpt2, execution_device=0)
        outputs = gpt2.generate(inputs["input_ids"])
        assert (
            tokenizer.decode(outputs[0].tolist())
            == "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"
        )

    def test_disk_offload(self):
        model = ModelForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        with TemporaryDirectory() as tmp_dir:
            disk_offload(model, tmp_dir, execution_device=device)
            output = model(x)
            assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

            # Clean up for next test.
            remove_hook_from_submodules(model)

        with TemporaryDirectory() as tmp_dir:
            disk_offload(model, tmp_dir, execution_device=device, offload_buffers=True)
            output = model(x)
            assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

    def test_disk_offload_with_unused_submodules(self):
        model = ModelWithUnusedSubModulesForTest()
        x = torch.randn(2, 3)
        expected = model(x)

        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        with TemporaryDirectory() as tmp_dir:
            disk_offload(
                model, tmp_dir, execution_device=device, preload_module_classes=["ModuleWithUnusedSubModules"]
            )
            output = model(x)
            assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

            # Clean up for next test.
            remove_hook_from_submodules(model)

        with TemporaryDirectory() as tmp_dir:
            disk_offload(
                model,
                tmp_dir,
                execution_device=device,
                offload_buffers=True,
                preload_module_classes=["ModuleWithUnusedSubModules"],
            )
            output = model(x)
            assert torch.allclose(expected, output.cpu(), 1e-4, 1e-5), f"Expected: {expected}, Actual: {output.cpu()}"

    @slow
    @require_cuda
    def test_disk_offload_gpt2(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello world! My name is", return_tensors="pt").to(0)

        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        with TemporaryDirectory() as tmp_dir:
            disk_offload(gpt2, tmp_dir, execution_device=0)
            outputs = gpt2.generate(inputs["input_ids"])
            assert (
                tokenizer.decode(outputs[0].tolist())
                == "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"
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
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_with_non_persistent_buffers(self):
        model = ModelForTestNonPersistentBuffers()
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": "disk"}
        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir, offload_buffers=True)
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_mps
    def test_dispatch_model_mps(self):
        model = ModelForTest()
        device_map = {"linear1": "mps", "batchnorm": "disk", "linear2": "disk"}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_tied_weights(self):
        model = ModelForTestTiedWeights()
        model.linear1.weight = model.linear2.weight
        device_map = {"linear1": 0, "batchnorm": 0, "linear2": 0}

        dispatch_model(model, device_map)
        assert model.linear2.weight is model.linear1.weight

    @require_multi_gpu
    def test_dispatch_model_tied_weights_memory(self):
        # Test that we do not duplicate tied weights at any point during dispatch_model call.

        torch.cuda.empty_cache()  # Needed in case we run several tests in a row.

        model = nn.Sequential(
            OrderedDict(
                [
                    ("linear0", nn.Linear(5000, 5000, bias=False)),
                    ("linear1", nn.Linear(5000, 5000, bias=False)),
                    ("linear2", nn.Linear(5000, 5000, bias=False)),
                    ("linear3", nn.Linear(5000, 5000, bias=False)),
                    ("linear4", nn.Linear(5000, 5000, bias=False)),
                ]
            )
        )
        model.linear2.weight = model.linear0.weight
        model.linear3.weight = model.linear0.weight
        model.linear4.weight = model.linear0.weight

        x = torch.randn(5, 5000)
        with torch.no_grad():
            expected = model(x)

        # We should need only 5000 * 5000 * 32 // 8 * 1e-6 = 100 MB on the device 0 for the four linear weights.
        device_map = {"linear0": 0, "linear1": 1, "linear2": 0, "linear3": 0, "linear4": 0}

        # Just to intialize CUDA context.
        a = torch.rand(5).to("cuda:0")  # noqa: F841

        free_memory_bytes = torch.cuda.mem_get_info("cuda:0")[0]
        required_memory_bytes = 5000 * 5000 * (32 // 8)

        # Leaving 50 MB of free memory for possible buffers, etc.
        n_vals = (free_memory_bytes - required_memory_bytes - int(50e6)) // (32 // 8)
        foo = torch.rand(n_vals, device="cuda:0")  # noqa: F841

        # If this does OOM: there is an issue in somewhere in dispatch_model, memory of tied weights is duplicated.
        try:
            dispatch_model(model, device_map)
        except torch.cuda.OutOfMemoryError as e:
            raise torch.cuda.OutOfMemoryError(
                f"OOM error in dispatch_model. This is a bug and should not happen, see test_dispatch_model_tied_weights_memory. {e}"
            )
        except Exception as e:
            raise e

        with torch.no_grad():
            output = model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_tied_weights_memory_with_nested_offload_cpu(self):
        # Test that we do not duplicate tied weights at any point during dispatch_model call.

        torch.cuda.empty_cache()  # Needed in case we run several tests in a row.

        class SubModule(torch.nn.Module):
            def __init__(self, ref_to_parameter):
                super().__init__()
                self.parameter = ref_to_parameter

            def forward(self, x):
                return x + torch.max(self.parameter)

        class LinearModuleAndSubModule(torch.nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features, bias=False)
                self.weight_submodule = SubModule(self.weight)
                self.weight_submodule2 = SubModule(self.weight)
                self.weight_submodule3 = SubModule(self.weight)
                self.weight_submodule4 = SubModule(self.weight)

            def forward(self, x):
                a = torch.nn.functional.linear(self.weight_submodule(x), self.weight)
                b = torch.nn.functional.linear(self.weight_submodule2(x), self.weight)
                c = torch.nn.functional.linear(self.weight_submodule3(x), self.weight)
                d = torch.nn.functional.linear(self.weight_submodule4(x), self.weight)
                return a + b + c + d

        class ModelWithSubmodules(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.compute = LinearModuleAndSubModule(5000, 5000)
                self.compute1 = LinearModuleAndSubModule(5000, 5000)

            def forward(self, x):
                a = self.compute(x)
                b = self.compute1(x)
                return a + b

        # We should need only 2 * 5000 * 5000 * 32 // 8 * 1e-6 = 200 MB on the device 0 for the whole model forward, and not 600 MB.
        device_map = {"compute": 0, "compute1": "cpu"}

        model = ModelWithSubmodules()

        x = torch.randn(1, 5000)
        with torch.no_grad():
            expected = model(x)

        # Just to intialize CUDA context.
        a = torch.rand(5).to("cuda:0")  # noqa: F841

        free_memory_bytes = torch.cuda.mem_get_info("cuda:0")[0]
        required_memory_bytes = 2 * 5000 * 5000 * (32 // 8)  # 200 MB

        # Leaving 150 MB of free memory for possible buffers, etc.
        n_vals = (free_memory_bytes - required_memory_bytes - int(150e6)) // (32 // 8)
        foo = torch.rand(n_vals, device="cuda:0")  # noqa: F841

        free_memory_bytes_before_dispatch = torch.cuda.mem_get_info("cuda:0")[0]
        dispatch_model(model, device_map)
        free_memory_bytes_after_dispatch = torch.cuda.mem_get_info("cuda:0")[0]

        assert (free_memory_bytes_after_dispatch - free_memory_bytes_before_dispatch) * 1e-6 < 130

        original_pointer = model.compute1._hf_hook.weights_map["weight"].data_ptr()

        with torch.no_grad():
            try:
                output = model(x)
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(
                    f"OOM error in dispatch_model. This is a bug and should not happen, see test_dispatch_model_tied_weights_memory_with_nested_offload_cpu. {e}"
                )
            except Exception as e:
                raise e

        assert torch.allclose(expected, output.cpu(), atol=1e-5)

        torch.cuda.empty_cache()

        free_memory_bytes_after_infer = torch.cuda.mem_get_info("cuda:0")[0]

        # Check that we have no more references on GPU for the offloaded tied weight.
        assert len(model.compute1.weight_submodule._hf_hook.tied_params_map[original_pointer]) == 0
        assert len(model.compute1._hf_hook.tied_params_map[original_pointer]) == 0
        assert (free_memory_bytes_after_infer - free_memory_bytes_after_dispatch) * 1e-6 < 130

        # Test is flacky otherwise.
        del model
        gc.collect()

    @require_cuda
    def test_dispatch_model_tied_weights_memory_with_nested_offload_disk(self):
        # Test that we do not duplicate tied weights at any point during dispatch_model call.

        torch.cuda.empty_cache()  # Needed in case we run several tests in a row.

        class SubModule(torch.nn.Module):
            def __init__(self, ref_to_parameter):
                super().__init__()
                self.parameter = ref_to_parameter

            def forward(self, x):
                return x + torch.max(self.parameter)

        class LinearModuleAndSubModule(torch.nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features, bias=False)
                self.weight_submodule = SubModule(self.weight)
                self.weight_submodule2 = SubModule(self.weight)
                self.weight_submodule3 = SubModule(self.weight)
                self.weight_submodule4 = SubModule(self.weight)

            def forward(self, x):
                a = torch.nn.functional.linear(self.weight_submodule(x), self.weight)
                b = torch.nn.functional.linear(self.weight_submodule2(x), self.weight)
                c = torch.nn.functional.linear(self.weight_submodule3(x), self.weight)
                d = torch.nn.functional.linear(self.weight_submodule4(x), self.weight)
                return a + b + c + d

        class ModelWithSubmodules(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.compute = LinearModuleAndSubModule(5000, 5000)
                self.compute1 = LinearModuleAndSubModule(5000, 5000)

            def forward(self, x):
                a = self.compute(x)
                b = self.compute1(x)
                return a + b

        # We should need only 2 * 5000 * 5000 * 32 // 8 * 1e-6 = 200 MB on the device 0 for the whole model forward, and not 600 MB.
        device_map = {"compute": 0, "compute1": "disk"}

        model = ModelWithSubmodules()

        x = torch.randn(1, 5000)
        with torch.no_grad():
            expected = model(x)

        # Just to intialize CUDA context.
        a = torch.rand(5).to("cuda:0")  # noqa: F841

        free_memory_bytes = torch.cuda.mem_get_info("cuda:0")[0]
        required_memory_bytes = 2 * 5000 * 5000 * (32 // 8)  # 200 MB

        # Leaving 150 MB of free memory for possible buffers, etc.
        n_vals = (free_memory_bytes - required_memory_bytes - int(200e6)) // (32 // 8)
        foo = torch.rand(n_vals, device="cuda:0")  # noqa: F841

        free_memory_bytes_before_dispatch = torch.cuda.mem_get_info("cuda:0")[0]
        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            free_memory_bytes_after_dispatch = torch.cuda.mem_get_info("cuda:0")[0]

            assert (free_memory_bytes_after_dispatch - free_memory_bytes_before_dispatch) * 1e-6 < 130

            with torch.no_grad():
                try:
                    output = model(x)
                except torch.cuda.OutOfMemoryError as e:
                    raise torch.cuda.OutOfMemoryError(
                        f"OOM error in dispatch_model. This is a bug and should not happen, see test_dispatch_model_tied_weights_memory_with_nested_offload_disk. {e}"
                    )
                except Exception as e:
                    raise e

            assert torch.allclose(expected, output.cpu(), atol=1e-5)

            torch.cuda.empty_cache()

            free_memory_bytes_after_infer = torch.cuda.mem_get_info("cuda:0")[0]

            # Check that we have no more references on GPU for the offloaded tied weight.
            n_non_empty = 0
            for pointer, pointer_dict in model.compute1.weight_submodule._hf_hook.tied_params_map.items():
                if len(pointer_dict) > 0:
                    n_non_empty += 1
            assert n_non_empty == 1  # `compute` layer one.

            n_non_empty = 0
            for pointer, pointer_dict in model.compute1._hf_hook.tied_params_map.items():
                if len(pointer_dict) > 0:
                    n_non_empty += 1
            assert n_non_empty == 1  # `compute` layer one.

            assert (free_memory_bytes_after_infer - free_memory_bytes_after_dispatch) * 1e-6 < 130

    @require_multi_gpu
    def test_dispatch_model_multi_gpu(self):
        model = BiggerModelForTest()
        device_map = {"linear1": "cpu", "linear2": "disk", "batchnorm": "cpu", "linear3": 0, "linear4": 1}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_copy(self):
        original_model = ModelForTestCopy(id=1)
        device_map = {"linear1": 0, "batchnorm": "cpu", "linear2": 0}

        x = torch.randn(2, 3)
        expected, original_output_id = original_model(x)

        dispatch_model(original_model, device_map)

        copied_model = copy.deepcopy(original_model)
        copied_model.id = 2
        output, copied_output_id = copied_model(x)

        assert original_model.id == original_output_id
        assert copied_model.id == copied_output_id
        assert copied_model.linear1.forward is not original_model.linear1.forward
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_move_offloaded_model(self):
        model = ModelForTest()
        device_map = {"linear1": "disk", "batchnorm": "cpu", "linear2": 0}
        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            with self.assertRaises(RuntimeError):
                model.to(0)

    @require_multi_gpu
    def test_dispatch_model_move_model_warning(self):
        model = ModelForTest()
        device_map = {"linear1": 0, "batchnorm": 0, "linear2": 1}
        with TemporaryDirectory() as tmp_dir:
            dispatch_model(model, device_map, offload_dir=tmp_dir)
            with self.assertLogs("accelerate.big_modeling", level="WARNING"):
                model.to("cpu")
            with self.assertLogs("accelerate.big_modeling", level="WARNING"):
                model.cuda(0)
            with self.assertRaises(RuntimeError):
                x = torch.randn(2, 3)
                model(x)

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
            "lm_head": 0,
        }
        for i in range(12):
            device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

        gpt2 = dispatch_model(gpt2, device_map)
        outputs = gpt2.generate(inputs["input_ids"])
        assert (
            tokenizer.decode(outputs[0].tolist())
            == "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"
        )

        # Dispatch with a bit of CPU offload
        gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        for i in range(4):
            device_map[f"transformer.h.{i}"] = "cpu"
        gpt2 = dispatch_model(gpt2, device_map)
        outputs = gpt2.generate(inputs["input_ids"])
        assert (
            tokenizer.decode(outputs[0].tolist())
            == "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"
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
            assert (
                tokenizer.decode(outputs[0].tolist())
                == "Hello world! My name is Kiyoshi, and I'm a student at the University of Tokyo"
            )

    @require_cuda
    def test_dispatch_model_with_unused_submodules(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "cpu", "linear2": "disk", "batchnorm": "cpu", "linear3": 0, "linear4": 0}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(
                model, device_map, offload_dir=tmp_dir, preload_module_classes=["ModuleWithUnusedSubModules"]
            )
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_mps
    def test_dispatch_model_with_unused_submodules_mps(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "mps", "linear2": "mps", "batchnorm": "mps", "linear3": "mps", "linear4": "disk"}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(
                model, device_map, offload_dir=tmp_dir, preload_module_classes=["ModuleWithUnusedSubModules"]
            )
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_multi_gpu
    def test_dispatch_model_with_unused_submodules_multi_gpu(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "cpu", "linear2": "disk", "batchnorm": "cpu", "linear3": 0, "linear4": 1}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            dispatch_model(
                model, device_map, offload_dir=tmp_dir, preload_module_classes=["ModuleWithUnusedSubModules"]
            )
            output = model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_dispatch_model_force_hooks(self):
        model = ModelForTest()
        device_map = {"": 0}

        x = torch.randn(2, 3)
        expected = model(x)

        dispatch_model(model, device_map, force_hooks=True)
        output = model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_load_checkpoint_and_dispatch(self):
        model = ModelForTest()
        device_map = {"linear1": "cpu", "batchnorm": "cpu", "linear2": 0}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = ModelForTest()
            new_model = load_checkpoint_and_dispatch(new_model, checkpoint, device_map=device_map)

        # CPU-offloaded weights are on the meta device while waiting for the forward pass.
        assert new_model.linear1.weight.device == torch.device("meta")
        assert new_model.linear2.weight.device == torch.device(0)

        output = new_model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_mps
    def test_load_checkpoint_and_dispatch_mps(self):
        model = ModelForTest()
        device_map = {"linear1": "mps", "batchnorm": "mps", "linear2": "disk"}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = ModelForTest()
            new_model = load_checkpoint_and_dispatch(
                new_model, checkpoint, device_map=device_map, offload_folder=tmp_dir
            )

            # CPU-offloaded weights are on the meta device while waiting for the forward pass.
            assert new_model.linear1.weight.device == torch.device("mps:0")
            assert new_model.linear2.weight.device == torch.device("meta")

            output = new_model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_multi_gpu
    def test_load_checkpoint_and_dispatch_multi_gpu(self):
        model = BiggerModelForTest()
        device_map = {"linear1": "cpu", "linear2": "cpu", "batchnorm": 0, "linear3": 0, "linear4": 1}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = BiggerModelForTest()
            new_model = load_checkpoint_and_dispatch(new_model, checkpoint, device_map=device_map)

        # CPU-offloaded weights are on the meta device while waiting for the forward pass.
        assert new_model.linear1.weight.device == torch.device("meta")
        assert new_model.linear2.weight.device == torch.device("meta")
        assert new_model.linear3.weight.device == torch.device(0)
        assert new_model.linear4.weight.device == torch.device(1)

        output = new_model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_load_checkpoint_and_dispatch_with_unused_submodules(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "cpu", "linear2": "cpu", "batchnorm": 0, "linear3": 0, "linear4": 0}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = ModelWithUnusedSubModulesForTest()
            new_model = load_checkpoint_and_dispatch(
                new_model, checkpoint, device_map=device_map, preload_module_classes=["ModuleWithUnusedSubModules"]
            )

        # CPU-offloaded weights are on the meta device while waiting for the forward pass.
        assert new_model.linear1.linear.weight.device == torch.device("meta")
        assert new_model.linear2.linear.weight.device == torch.device("meta")
        assert new_model.linear3.linear.weight.device == torch.device(0)
        assert new_model.linear4.linear.weight.device == torch.device(0)

        output = new_model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_mps
    def test_load_checkpoint_and_dispatch_with_unused_submodules_mps(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "mps", "linear2": "mps", "batchnorm": "mps", "linear3": "disk", "linear4": "disk"}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = ModelWithUnusedSubModulesForTest()
            new_model = load_checkpoint_and_dispatch(
                new_model,
                checkpoint,
                device_map=device_map,
                preload_module_classes=["ModuleWithUnusedSubModules"],
                offload_folder=tmp_dir,
            )

            # CPU-offloaded weights are on the meta device while waiting for the forward pass.
            assert new_model.linear1.linear.weight.device == torch.device("mps:0")
            assert new_model.linear2.linear.weight.device == torch.device("mps:0")
            assert new_model.linear3.linear.weight.device == torch.device("meta")
            assert new_model.linear4.linear.weight.device == torch.device("meta")

            output = new_model(x)
            assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_multi_gpu
    def test_load_checkpoint_and_dispatch_multi_gpu_with_unused_submodules(self):
        model = ModelWithUnusedSubModulesForTest()
        device_map = {"linear1": "cpu", "linear2": "cpu", "batchnorm": 0, "linear3": 0, "linear4": 1}

        x = torch.randn(2, 3)
        expected = model(x)

        with TemporaryDirectory() as tmp_dir:
            checkpoint = os.path.join(tmp_dir, "pt_model.bin")
            torch.save(model.state_dict(), checkpoint)

            new_model = ModelWithUnusedSubModulesForTest()
            new_model = load_checkpoint_and_dispatch(
                new_model, checkpoint, device_map=device_map, preload_module_classes=["ModuleWithUnusedSubModules"]
            )

        # CPU-offloaded weights are on the meta device while waiting for the forward pass.
        assert new_model.linear1.linear.weight.device == torch.device("meta")
        assert new_model.linear2.linear.weight.device == torch.device("meta")
        assert new_model.linear3.linear.weight.device == torch.device(0)
        assert new_model.linear4.linear.weight.device == torch.device(1)

        output = new_model(x)
        assert torch.allclose(expected, output.cpu(), atol=1e-5)

    @require_cuda
    def test_cpu_offload_with_hook(self):
        model1 = torch.nn.Linear(4, 5)
        model1, hook1 = cpu_offload_with_hook(model1)
        assert model1.weight.device == torch.device("cpu")

        inputs = torch.randn(3, 4)
        outputs = model1(inputs)
        assert outputs.device == torch.device(0)
        assert model1.weight.device == torch.device(0)

        hook1.offload()
        assert model1.weight.device == torch.device("cpu")

        model2 = torch.nn.Linear(5, 5)
        model2, hook2 = cpu_offload_with_hook(model2, prev_module_hook=hook1)
        assert model2.weight.device == torch.device("cpu")

        outputs = model1(inputs)
        assert outputs.device == torch.device(0)
        assert model1.weight.device == torch.device(0)

        outputs = model2(outputs)
        assert outputs.device == torch.device(0)
        assert model1.weight.device == torch.device("cpu")
        assert model2.weight.device == torch.device(0)

        hook2.offload()
        assert model2.weight.device == torch.device("cpu")

    @require_non_torch_xla
    @slow
    @require_bnb
    @require_multi_gpu
    def test_dispatch_model_bnb(self):
        """Tests that `dispatch_model` quantizes int8 layers"""
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
        from transformers.utils.bitsandbytes import replace_with_bnb_linear

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map="balanced",
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.int8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

        assert model.h[(-1)].self_attention.query_key_value.weight.dtype == torch.int8
        assert model.h[(-1)].self_attention.query_key_value.weight.device.index == 1

    @require_non_torch_xla
    @slow
    @require_bnb
    def test_dispatch_model_int8_simple(self):
        """Tests that `dispatch_model` quantizes int8 layers"""
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
        from transformers.utils.bitsandbytes import replace_with_bnb_linear

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

        # test with auto
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map="auto",
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.int8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        # test with str device map
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map={"": torch.device("cuda:0")},
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.int8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        # test with torch.device device map
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map={"": "cuda:0"},
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.int8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

    @require_non_torch_xla
    @slow
    @require_bnb
    def test_dipatch_model_fp4_simple(self):
        """Tests that `dispatch_model` quantizes fp4 layers"""
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
        from transformers.utils.bitsandbytes import replace_with_bnb_linear

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

        # test with auto
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map="auto",
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.uint8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        # test with str device map
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map={"": torch.device("cuda:0")},
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.uint8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0

        with init_empty_weights():
            model = AutoModel.from_config(AutoConfig.from_pretrained("bigscience/bloom-560m"))

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=["lm_head"], quantization_config=quantization_config
        )

        # test with torch.device device map
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map={"": "cuda:0"},
        )

        assert model.h[0].self_attention.query_key_value.weight.dtype == torch.uint8
        assert model.h[0].self_attention.query_key_value.weight.device.index == 0
