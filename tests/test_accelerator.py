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
import itertools
import json
import os
import pickle
import tempfile
import time
from unittest.mock import patch

import psutil
import torch
from parameterized import parameterized
from torch.utils.data import DataLoader, TensorDataset

from accelerate import DistributedType, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.accelerator import Accelerator
from accelerate.data_loader import DataLoaderDispatcher, DataLoaderShard, skip_first_batches
from accelerate.state import GradientState, PartialState
from accelerate.test_utils import (
    require_bnb,
    require_cuda_or_xpu,
    require_fp8,
    require_fp16,
    require_huggingface_suite,
    require_multi_device,
    require_non_cpu,
    require_non_hpu,
    require_transformer_engine,
    slow,
    torch_device,
)
from accelerate.test_utils.testing import (
    AccelerateTestCase,
    require_cuda,
    require_non_torch_xla,
    require_torchdata_stateful_dataloader,
)
from accelerate.utils import FP8RecipeKwargs, is_torchdata_stateful_dataloader_available, patch_environment
from accelerate.utils.dataclasses import DataLoaderConfiguration
from accelerate.utils.modeling import get_state_dict_from_offload, load_checkpoint_in_model
from accelerate.utils.random import set_seed


if is_torchdata_stateful_dataloader_available():
    from torchdata.stateful_dataloader import StatefulDataLoader


class ModelWithTiedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear2.weight = self.linear1.weight
        self.linear2.bias = self.linear1.bias

    def forward(self, x):
        return self.linear2(self.linear1(x))


def create_components(tied_weights=False):
    model = ModelWithTiedWeights() if tied_weights else torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    train_dl = DataLoader(TensorDataset(torch.tensor([1, 2, 3])))
    valid_dl = DataLoader(TensorDataset(torch.tensor([4, 5, 6])))
    return model, optimizer, scheduler, train_dl, valid_dl


class ModelForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.batchnorm = torch.nn.BatchNorm1d(4)
        self.linear2 = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


def create_dataloaders_for_test(batch_size=3, n_train_batches: int = 12, n_valid_batches: int = 2, num_workers=0):
    "Generates a tuple of dummy DataLoaders to test with"

    def get_dataset(n_batches):
        x = torch.randn(batch_size * n_batches, 3)
        y = torch.randn(batch_size * n_batches, 5)
        return TensorDataset(x, y)

    train_dataset = get_dataset(n_train_batches)
    valid_dataset = get_dataset(n_valid_batches)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    return (train_dataloader, valid_dataloader)


def get_signature(model):
    return sum(param.abs().sum().item() for param in model.parameters())


def load_random_weights(model):
    if isinstance(model, torch.nn.Linear):
        state = torch.nn.Linear(*tuple(model.weight.T.shape)).state_dict()
    elif isinstance(model, ModelWithTiedWeights):
        state = ModelWithTiedWeights().state_dict()
    model.load_state_dict(state)


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = "use_safetensors" if param.args[0] is True else "use_pytorch"
    if len(param.args) > 1:
        param_based_name += "_tied_weights" if param.args[1] is True else ""
    if len(param.args) > 2:
        param_based_name += f"_num_workers_{param.args[2]}"
    if len(param.args) > 3:
        param_based_name += "_dispatch_batches" if param.args[3] is True else "_no_dispatch_batches"
    return f"{func.__name__}_{param_based_name}"


class AcceleratorTester(AccelerateTestCase):
    def test_partial_state_after_reset(self):
        # Verifies that custom getattr errors will be thrown
        # if the state is reset, but only if trying to
        # get expected attributes
        state = PartialState()
        assert state.num_processes > 0

        with self.assertRaises(AttributeError) as cm:
            state.someotherthing
        assert "'PartialState' object has no attribute" in str(cm.exception)
        assert "This happens if `PartialState._reset_state()`" not in str(cm.exception)

        with self.assertRaises(AttributeError) as cm:
            state._reset_state()
            state.num_processes
        assert "`PartialState` object has no attribute" in str(cm.exception)
        assert "This happens if `PartialState._reset_state()`" in str(cm.exception)

        state.someotherthing = "MyValue"
        assert state.someotherthing == "MyValue"

    def test_accelerator_state_after_reset(self):
        # Verifies that custom getattr errors will be thrown
        # if the state is reset, but only if trying to
        # get expected attributes
        accelerator = Accelerator()
        assert accelerator.num_processes > 0

        with self.assertRaises(AttributeError) as cm:
            accelerator.state.someotherthing
        assert "'AcceleratorState' object has no attribute" in str(cm.exception)
        assert "This happens if `AcceleratorState._reset_state()`" not in str(cm.exception)

        with self.assertRaises(AttributeError) as cm:
            accelerator.state._reset_state()
            accelerator.num_processes
        assert "`AcceleratorState` object has no attribute" in str(cm.exception)
        assert "This happens if `AcceleratorState._reset_state()`" in str(cm.exception)

        accelerator.state.someotherthing = "MyValue"
        assert accelerator.state.someotherthing == "MyValue"

    @require_non_cpu
    def test_accelerator_can_be_reinstantiated(self):
        _ = Accelerator()
        assert PartialState._shared_state["_cpu"] is False
        assert PartialState._shared_state["device"].type in ["cuda", "mps", "npu", "xpu", "xla", "hpu"]
        with self.assertRaises(ValueError):
            _ = Accelerator(cpu=True)

    @require_cuda
    def test_setting_cpu_affinity(self):
        with patch_environment(accelerate_cpu_affinity=1, accelerate_debug_mode=1):
            with self.assertLogs("accelerate.utils.environment", level="INFO") as cm:
                _ = Accelerator()
                assert any("Assigning" in log for log in cm.output)
                assert any("cpu cores to process" in log for log in cm.output)

    def test_mutable_states(self):
        accelerator = Accelerator()
        state = GradientState()
        assert state.num_steps == 1
        accelerator.gradient_accumulation_steps = 4
        assert state.num_steps == 4

        assert state.sync_gradients is True
        accelerator.sync_gradients = False
        assert state.sync_gradients is False
        GradientState._reset_state()

    def test_prepared_objects_are_referenced(self):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()

        (
            prepared_model,
            prepared_optimizer,
            prepared_scheduler,
            prepared_train_dl,
            prepared_valid_dl,
        ) = accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        assert prepared_model in accelerator._models
        assert prepared_optimizer in accelerator._optimizers
        assert prepared_scheduler in accelerator._schedulers
        assert prepared_train_dl in accelerator._dataloaders
        assert prepared_valid_dl in accelerator._dataloaders

    @require_non_hpu  # hpu does not support empty_cache
    def test_free_memory_dereferences_prepared_components(self):
        accelerator = Accelerator()
        # Free up refs with empty_cache() and gc.collect()
        accelerator.free_memory()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        free_cpu_ram_before = psutil.virtual_memory().available // 1024 // 1024
        model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
            model, optimizer, scheduler, train_dl, valid_dl
        )

        # Short sleep here makes this test more reliable
        time.sleep(1e-3)

        model, optimizer, scheduler, train_dl, valid_dl = accelerator.free_memory(
            model, optimizer, scheduler, train_dl, valid_dl
        )

        free_cpu_ram_after = psutil.virtual_memory().available // 1024 // 1024

        assert len(accelerator._models) == 0
        assert len(accelerator._optimizers) == 0
        assert len(accelerator._schedulers) == 0
        assert len(accelerator._dataloaders) == 0

        # The less-than comes *specifically* from CUDA CPU things/won't be present on CPU builds
        assert free_cpu_ram_after <= free_cpu_ram_before

    @require_non_torch_xla
    def test_env_var_device(self):
        """Tests that setting the torch device with ACCELERATE_TORCH_DEVICE overrides default device."""
        PartialState._reset_state()

        # Mock torch.cuda.set_device to avoid an exception as the device doesn't exist
        def noop(*args, **kwargs):
            pass

        with patch("torch.cuda.set_device", noop), patch_environment(ACCELERATE_TORCH_DEVICE="cuda:64"):
            accelerator = Accelerator()
            assert str(accelerator.state.device) == "cuda:64"

    @parameterized.expand([(True, True), (True, False), (False, False)], name_func=parameterized_custom_name_func)
    def test_save_load_model(self, use_safetensors, tied_weights):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components(tied_weights)
        accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        model_signature = get_signature(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname, safe_serialization=use_safetensors)

            # make sure random weights don't match
            load_random_weights(model)
            assert abs(model_signature - get_signature(model)) > 1e-3

            # make sure loaded weights match
            accelerator.load_state(tmpdirname)
            assert abs(model_signature - get_signature(model)) < 1e-3

    @parameterized.expand([True, False], name_func=parameterized_custom_name_func)
    def test_save_model(self, use_safetensors):
        accelerator = Accelerator()
        model = torch.nn.Linear(10, 10)

        model_signature = get_signature(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_model(model, tmpdirname, safe_serialization=use_safetensors)
            # make sure loaded weights match
            load_checkpoint_in_model(model, tmpdirname)
            assert abs(model_signature - get_signature(model)) < 1e-3

    @parameterized.expand([True, False], name_func=parameterized_custom_name_func)
    def test_save_sharded_model(self, use_safetensors):
        accelerator = Accelerator()
        inputs = torch.randn(3, 3)
        model = ModelForTest()
        expected = model(inputs)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # By setting it to 100, we will split the model int 3 shards
            accelerator.save_model(model, tmpdirname, safe_serialization=use_safetensors, max_shard_size=100)
            # make sure loaded weights match
            load_checkpoint_in_model(model, tmpdirname)
            output = model(inputs)

        assert torch.allclose(expected, output, atol=1e-5)

    @parameterized.expand([True, False], name_func=parameterized_custom_name_func)
    def test_save_model_offload(self, use_safetensors):
        accelerator = Accelerator()

        device_map = {"linear1": "cpu", "batchnorm": "disk", "linear2": "cpu"}

        inputs = torch.randn(3, 3)
        model = ModelForTest()
        expected = model(inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            accelerator.save_model(model, tmp_dir, safe_serialization=use_safetensors)
            # load and save offloaded model
            load_checkpoint_and_dispatch(model, tmp_dir, device_map=device_map, offload_folder=tmp_dir)
            accelerator.save_model(model, tmp_dir, safe_serialization=use_safetensors)

            # load weights that were saved from the offloaded model
            load_checkpoint_and_dispatch(model, tmp_dir)
            output = model(inputs)
        assert torch.allclose(expected, output, atol=1e-5)

    @parameterized.expand([True, False], name_func=parameterized_custom_name_func)
    @require_non_cpu
    def test_get_state_dict_from_offload(self, use_safetensors):
        accelerator = Accelerator()

        device_map = {"linear1": "cpu", "batchnorm": "disk", "linear2": "disk"}
        model = ModelForTest()
        offloaded_layer_weight = model.linear2.weight
        with tempfile.TemporaryDirectory() as tmp_dir:
            accelerator.save_model(model, tmp_dir, safe_serialization=use_safetensors)
            # load model with offloaded layers
            load_checkpoint_and_dispatch(model, tmp_dir, device_map=device_map, offload_folder=tmp_dir)
            cpu_onloaded_layer = get_state_dict_from_offload(
                model.linear2, "linear2.weight", {"linear2.weight": ""}, device_to_put_offload="cpu"
            )
            device_onloaded_layer = get_state_dict_from_offload(
                model.linear2, "linear2.weight", {"linear2.weight": ""}, device_to_put_offload=0
            )
            cpu_onloaded_layer_weight = cpu_onloaded_layer["linear2.weight"]
            device_onloaded_layer_weight = device_onloaded_layer["linear2.weight"]

        assert torch.allclose(offloaded_layer_weight, cpu_onloaded_layer_weight)
        assert torch.allclose(
            offloaded_layer_weight, device_onloaded_layer_weight.to("cpu")
        )  # must be on the same device for torch.allclose()
        assert cpu_onloaded_layer_weight.device.type == "cpu"
        assert device_onloaded_layer_weight.device.type == torch_device

    @parameterized.expand([True, False], name_func=parameterized_custom_name_func)
    def test_save_load_model_with_hooks(self, use_safetensors):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        model_signature = get_signature(model)

        # saving hook
        def save_config(models, weights, output_dir):
            config = {"class_name": models[0].__class__.__name__}

            with open(os.path.join(output_dir, "data.json"), "w") as f:
                json.dump(config, f)

        # loading hook
        def load_config(models, input_dir):
            with open(os.path.join(input_dir, "data.json")) as f:
                config = json.load(f)

            models[0].class_name = config["class_name"]

        save_hook = accelerator.register_save_state_pre_hook(save_config)
        load_hook = accelerator.register_load_state_pre_hook(load_config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname, safe_serialization=use_safetensors)

            # make sure random weights don't match with hooks
            load_random_weights(model)
            assert abs(model_signature - get_signature(model)) > 1e-3

            # random class name to verify correct one is loaded
            model.class_name = "random"

            # make sure loaded weights match with hooks
            accelerator.load_state(tmpdirname)
            assert abs(model_signature - get_signature(model)) < 1e-3

            # mode.class_name is loaded from config
            assert model.class_name == model.__class__.__name__

        # remove hooks
        save_hook.remove()
        load_hook.remove()

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname, safe_serialization=use_safetensors)

            # make sure random weights don't match with hooks removed
            load_random_weights(model)
            assert abs(model_signature - get_signature(model)) > 1e-3

            # random class name to verify correct one is loaded
            model.class_name = "random"

            # make sure loaded weights match with hooks removed
            accelerator.load_state(tmpdirname)
            assert abs(model_signature - get_signature(model)) < 1e-3

            # mode.class_name is NOT loaded from config
            assert model.class_name != model.__class__.__name__

    def test_accelerator_none(self):
        """Just test that passing None to accelerator.prepare() works."""
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        dummy_obj = None

        # This should work
        model, optimizer, scheduler, train_dl, valid_dl, dummy_obj = accelerator.prepare(
            model, optimizer, scheduler, train_dl, valid_dl, dummy_obj
        )
        assert dummy_obj is None

    def test_is_accelerator_prepared(self):
        """Checks that `_is_accelerator_prepared` is set properly"""
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        dummy_obj = [1, 2, 3]

        # This should work
        model, optimizer, scheduler, train_dl, valid_dl, dummy_obj = accelerator.prepare(
            model, optimizer, scheduler, train_dl, valid_dl, dummy_obj
        )
        assert getattr(dummy_obj, "_is_accelerate_prepared", False) is False, (
            "Dummy object should have `_is_accelerate_prepared` set to `True`"
        )
        assert getattr(model, "_is_accelerate_prepared", False) is True, (
            "Model is missing `_is_accelerator_prepared` or is set to `False`"
        )
        assert getattr(optimizer, "_is_accelerate_prepared", False) is True, (
            "Optimizer is missing `_is_accelerator_prepared` or is set to `False`"
        )
        assert getattr(scheduler, "_is_accelerate_prepared", False) is True, (
            "Scheduler is missing `_is_accelerator_prepared` or is set to `False`"
        )
        assert getattr(train_dl, "_is_accelerate_prepared", False) is True, (
            "Train Dataloader is missing `_is_accelerator_prepared` or is set to `False`"
        )
        assert getattr(valid_dl, "_is_accelerate_prepared", False) is True, (
            "Valid Dataloader is missing `_is_accelerator_prepared` or is set to `False`"
        )

    @require_cuda_or_xpu
    @slow
    @require_bnb
    def test_accelerator_bnb(self):
        """Tests that the accelerator can be used with the BNB library."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125m",
            load_in_8bit=True,
            device_map={"": 0},
        )
        accelerator = Accelerator()

        # This should work
        model = accelerator.prepare(model)

    @require_cuda_or_xpu
    @slow
    @require_bnb
    def test_accelerator_bnb_cpu_error(self):
        """Tests that the accelerator can be used with the BNB library. This should fail as we are trying to load a model
        that is loaded between cpu and gpu"""
        from transformers import AutoModelForCausalLM

        accelerator = Accelerator()

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-125m",
            )
            model.tie_weights()
            device_map = infer_auto_device_map(model)
            device_map["lm_head"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125m", device_map=device_map, load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )

        # This should not work and get value error
        with self.assertRaises(ValueError):
            model = accelerator.prepare(model)

    @require_non_torch_xla
    @require_non_hpu  # bnb is not supported on HPU
    @slow
    @require_bnb
    @require_multi_device
    def test_accelerator_bnb_multi_device(self):
        """Tests that the accelerator can be used with the BNB library."""
        from transformers import AutoModelForCausalLM

        if torch_device == "cuda":
            PartialState._shared_state = {"distributed_type": DistributedType.MULTI_GPU}
        elif torch_device == "npu":
            PartialState._shared_state = {"distributed_type": DistributedType.MULTI_NPU}
        elif torch_device == "xpu":
            PartialState._shared_state = {"distributed_type": DistributedType.MULTI_XPU}
        else:
            raise ValueError(f"{torch_device} is not supported in test_accelerator_bnb_multi_device.")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-125m",
            )
            model.tie_weights()
            device_map = infer_auto_device_map(model)
            device_map["lm_head"] = 1

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125m",
            load_in_8bit=True,
            device_map=device_map,
        )
        accelerator = Accelerator()

        # This should not work and get value error
        with self.assertRaises(ValueError):
            _ = accelerator.prepare(model)

    @require_non_torch_xla
    @require_non_hpu  # bnb is not supported on HPU
    @slow
    @require_bnb
    @require_multi_device
    def test_accelerator_bnb_multi_device_no_distributed(self):
        """Tests that the accelerator can be used with the BNB library."""
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-neo-125m",
            )
            device_map = infer_auto_device_map(model)
            device_map["lm_head"] = 1

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-125m",
            load_in_8bit=True,
            device_map=device_map,
        )
        accelerator = Accelerator()

        # This should work
        _ = accelerator.prepare(model)

    @require_non_cpu
    def test_accelerator_cpu_flag_prepare(self):
        model = torch.nn.Linear(10, 10)
        sgd = torch.optim.SGD(model.parameters(), lr=0.01)
        accelerator = Accelerator(cpu=True)
        _ = accelerator.prepare(sgd)

    @require_fp8
    @require_transformer_engine
    def test_can_unwrap_model_te(self):
        model, optimizer, *_ = create_components()
        fp8_recipe = FP8RecipeKwargs(backend="TE")
        accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[fp8_recipe])
        inputs = torch.randn(10, 2).to(torch_device)
        model, optimizer = accelerator.prepare(model, optimizer)
        model(inputs)  # sanity check that this works

        model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        model(inputs)  # check that this still works

        # check that pickle roundtrip works
        model_loaded = pickle.loads(pickle.dumps(model))
        model_loaded(inputs)

    @require_fp16
    @require_non_cpu
    def test_can_unwrap_model_fp16(self):
        # test for a regression introduced in #872
        # before the fix, after unwrapping with keep_fp32_wrapper=False, there would be the following error:
        # Linear.forward() missing 1 required positional argument: 'input'
        model = create_components()[0]
        accelerator = Accelerator(mixed_precision="fp16")
        inputs = torch.randn(10, 2).to(torch_device)
        model = accelerator.prepare(model)
        model(inputs)  # sanity check that this works

        model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        model(inputs)  # check that this still works

        # check that pickle roundtrip works
        model_loaded = pickle.loads(pickle.dumps(model))
        model_loaded(inputs)

    def test_can_unwrap_model(self):
        model = create_components()[0]
        accelerator = Accelerator(mixed_precision="no", cpu=True)
        inputs = torch.randn(10, 2)
        model = accelerator.prepare(model)
        model(inputs)  # sanity check that this works

        model = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        model(inputs)  # check that this still works

        # check that pickle roundtrip works
        model_loaded = pickle.loads(pickle.dumps(model))
        model_loaded(inputs)

    def test_can_unwrap_distributed_compiled_model_keep_torch_compile(self):
        model = create_components()[0]
        accelerator = Accelerator()

        compiled_model = torch.compile(model)

        distributed_model = torch.nn.DataParallel(model)
        distributed_compiled_model = torch.compile(distributed_model)
        unwrapped_model = accelerator.unwrap_model(distributed_compiled_model, keep_torch_compile=True)

        assert compiled_model._orig_mod == unwrapped_model._orig_mod

    def test_can_unwrap_distributed_compiled_model_remove_torch_compile(self):
        model = create_components()[0]
        accelerator = Accelerator()

        compiled_model = torch.compile(model)

        distributed_model = torch.nn.DataParallel(model)
        distributed_compiled_model = torch.compile(distributed_model)
        unwrapped_model = accelerator.unwrap_model(distributed_compiled_model, keep_torch_compile=False)

        assert compiled_model._orig_mod == unwrapped_model

    @parameterized.expand([True, False])
    def test_can_pickle_dataloader(self, dispatch_batches):
        """
        Test that pickling a prepared dataloader works.
        """
        data = torch.arange(10).to(torch_device)
        ds = torch.utils.data.TensorDataset(data)
        dl = torch.utils.data.DataLoader(ds)
        skip_dl = skip_first_batches(dl, 2)

        # Currently, StatefulDataLoader doesn't seem to support pickling, so we aren't testing that functionality
        # TODO: Add support for pickling StatefulDataLoader
        dataloader_config = DataLoaderConfiguration(dispatch_batches=dispatch_batches, use_stateful_dataloader=False)
        accelerator = Accelerator(dataloader_config=dataloader_config)

        original_dl, _ = accelerator.prepare(dl, skip_dl)
        if dispatch_batches:
            assert isinstance(original_dl, DataLoaderDispatcher)
        else:
            assert isinstance(original_dl, DataLoaderShard)

        prepared_model_dumps = pickle.dumps(accelerator)

        model_loaded = pickle.loads(prepared_model_dumps)
        assert len(model_loaded._dataloaders) == 2

        # Assert equality of recovered and original dataloader
        loaded_dl = model_loaded._dataloaders[0]
        assert isinstance(loaded_dl, DataLoader)
        if dispatch_batches:
            assert isinstance(loaded_dl, DataLoaderDispatcher)
        else:
            assert isinstance(loaded_dl, DataLoaderShard)
        assert len(loaded_dl) == len(original_dl)
        assert [i for i in loaded_dl] == [i for i in original_dl]

        # Test skip dataloader works as expected as well
        loaded_skip_dl = model_loaded._dataloaders[1]
        assert isinstance(loaded_skip_dl, DataLoader)
        if dispatch_batches:
            assert isinstance(loaded_dl, DataLoaderDispatcher)
        else:
            assert isinstance(loaded_dl, DataLoaderShard)
        assert len(loaded_skip_dl) == len(original_dl) - 2
        assert [i for i in loaded_skip_dl] == [i for i in original_dl][2:]

    # Ideally would be a parameterized test which works with either stateful or non-stateful dataloaders, but dependencies are a bit awkward.
    @require_torchdata_stateful_dataloader
    def test_prepared_objects_are_referenced_with_stateful_dataloader(self):
        """Test that setting `use_stateful_dataloader=True` in `DataLoaderConfiguration` prepares a `StatefulDataLoader` object instead of a `DataLoader` object."""
        dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)
        accelerator = Accelerator(dataloader_config=dataloader_config)
        model, optimizer, scheduler, train_dl, valid_dl = create_components()

        (
            prepared_model,
            prepared_optimizer,
            prepared_scheduler,
            prepared_train_dl,
            prepared_valid_dl,
        ) = accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        assert prepared_model in accelerator._models
        assert prepared_optimizer in accelerator._optimizers
        assert prepared_scheduler in accelerator._schedulers
        assert prepared_train_dl in accelerator._dataloaders
        assert prepared_valid_dl in accelerator._dataloaders
        assert isinstance(prepared_train_dl, StatefulDataLoader)
        assert isinstance(prepared_valid_dl, StatefulDataLoader)

    @parameterized.expand(
        itertools.product([True, False], [True, False], [0, 2], [True, False]),
        name_func=parameterized_custom_name_func,
    )
    @require_torchdata_stateful_dataloader
    def test_save_model_with_stateful_dataloader(self, use_safetensors, tied_weights, num_workers, dispatch_batches):
        """
        Test that saving and loading a model with a stateful dataloader returns the same model,
        and that the dataloader's iterator is restored properly."""
        set_seed(42)
        n_train_batches = 64  # Use enough batches to ensure we can get partial iterations on large compute
        dataloader_config = DataLoaderConfiguration(dispatch_batches=dispatch_batches, use_stateful_dataloader=True)
        accelerator = Accelerator(dataloader_config=dataloader_config)

        model, optimizer, scheduler, train_dl, valid_dl = create_components(tied_weights)
        train_dl, valid_dl = create_dataloaders_for_test(n_train_batches=n_train_batches, num_workers=num_workers)
        model = ModelForTest()

        (
            prepared_model,
            prepared_optimizer,
            prepared_scheduler,
            prepared_train_dl,
            prepared_valid_dl,
        ) = accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        assert isinstance(prepared_train_dl, StatefulDataLoader)
        assert isinstance(prepared_valid_dl, StatefulDataLoader)

        # Perform 3 training iterations to ensure the dataloader's iterator is advanced
        num_batches_to_skip = 3
        model.train()
        untrained_batches = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for step, batch in enumerate(prepared_train_dl):
                x, y = batch
                outputs = prepared_model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
                accelerator.backward(loss)
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad()
                if step == num_batches_to_skip - 1:
                    # Save the state once we've gone through a few batches
                    accelerator.save_state(f"{tmpdirname}/state", safe_serialization=use_safetensors)
                if step >= num_batches_to_skip:
                    untrained_batches.append(batch)

            not_skipped_batches = accelerator.gather(untrained_batches)
            # We then unwrap the trained model
            unwrapped_model = accelerator.unwrap_model(prepared_model)

            original_linear1 = unwrapped_model.linear1.weight.clone()
            original_batchnorm = unwrapped_model.batchnorm.weight.clone()
            original_linear2 = unwrapped_model.linear2.weight.clone()

            # Resume the state
            accelerator.load_state(f"{tmpdirname}/state")

            # Train this to the end of the DataLoader
            batches_seen_with_loaded_dl = 0
            for batch in prepared_train_dl:
                x, y = batch
                outputs = prepared_model(x)
                loss = torch.nn.functional.mse_loss(outputs, y)
                accelerator.backward(loss)
                prepared_optimizer.step()
                prepared_scheduler.step()
                prepared_optimizer.zero_grad()
                batches_seen_with_loaded_dl += 1

            unwrapped_model_2 = accelerator.unwrap_model(prepared_model)

            new_linear1 = unwrapped_model_2.linear1.weight
            new_batchnorm = unwrapped_model_2.batchnorm.weight
            new_linear2 = unwrapped_model_2.linear2.weight

            # Assert equalities
            assert batches_seen_with_loaded_dl == len(not_skipped_batches)
            assert torch.allclose(original_linear1, new_linear1)
            assert torch.allclose(original_batchnorm, new_batchnorm)
            assert torch.allclose(original_linear2, new_linear2)

    @require_non_cpu
    @require_huggingface_suite
    def test_nested_hook(self):
        from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

        class MyLinear(torch.nn.Module):
            def __init__(self, device=None, dtype=None):
                factory_kwargs = {"device": device, "dtype": dtype}
                super().__init__()
                self.centroid = torch.nn.Embedding(1, 2)
                self.indices = torch.nn.Parameter(torch.empty((1, 2, 2), **factory_kwargs))

            def forward(self, x):
                orig_shape = x.shape
                x = torch.abs(x + self.indices).long()
                x = x % 2
                x = x.sum(-1)
                x = (self.centroid.weight + x).reshape(orig_shape)
                return x

        class MySubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = MyLinear()

            def forward(self, x):
                return self.layer(x)

        class MyModel(PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.layer = torch.nn.ModuleList([MySubModel() for i in range(4)])

            def forward(self, x):
                for layer in self.layer:
                    x = layer(x)
                return x

        with tempfile.TemporaryDirectory() as tmpdirname:
            check_point = tmpdirname
            offload_folder = check_point + "/offload"
            os.makedirs(offload_folder, exist_ok=True)
            config = PretrainedConfig()
            m = MyModel(config)
            m.save_pretrained(check_point)

            with init_empty_weights():
                my_model = MyModel(config)
            my_model = load_checkpoint_and_dispatch(
                my_model,
                checkpoint=check_point,
                max_memory={"cpu": 60, 0: 60},
                device_map="auto",
                no_split_module_classes=["MySubModel"],
                offload_folder=offload_folder,
                preload_module_classes=["MyLinear"],
            )
            # before fix, this would raise an error
            #       weight is on the meta device, we need a `value` to put in on 0
            x = torch.randn(1, 2)
            my_model(x)
