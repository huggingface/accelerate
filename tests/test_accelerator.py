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
import pickle
import tempfile
from unittest.mock import patch

import pytest
import torch
from parameterized import parameterized
from torch.utils.data import DataLoader, TensorDataset

from accelerate import DistributedType, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.accelerator import Accelerator
from accelerate.state import GradientState, PartialState
from accelerate.test_utils import require_bnb, require_multi_device, require_non_cpu, slow, torch_device
from accelerate.test_utils.testing import AccelerateTestCase, require_non_torch_xla
from accelerate.utils import patch_environment
from accelerate.utils.modeling import load_checkpoint_in_model


def create_components():
    model = torch.nn.Linear(2, 4)
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


def get_signature(model):
    return (model.weight.abs().sum() + model.bias.abs().sum()).item()


def load_random_weights(model):
    state = torch.nn.Linear(*tuple(model.weight.T.shape)).state_dict()
    model.load_state_dict(state)


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = "use_safetensors" if param.args[0] is True else "use_pytorch"
    return f"{func.__name__}_{param_based_name}"


class AcceleratorTester(AccelerateTestCase):
    # Should be removed after 1.0.0 release
    def test_deprecated_values(self):
        # Test defaults
        accelerator = Accelerator()
        assert accelerator.split_batches is False, "split_batches should be False by default"
        assert accelerator.dispatch_batches is None, "dispatch_batches should be None by default"
        assert accelerator.even_batches is True, "even_batches should be True by default"
        assert accelerator.use_seedable_sampler is False, "use_seedable_sampler should be False by default"

        # Pass some arguments only
        with pytest.warns(FutureWarning) as cm:
            accelerator = Accelerator(
                dispatch_batches=True,
                split_batches=False,
            )
            deprecation_warning = str(cm.list[0].message)
            assert accelerator.split_batches is False, "split_batches should be True"
            assert accelerator.dispatch_batches is True, "dispatch_batches should be True"
            assert accelerator.even_batches is True, "even_batches should be True by default"
            assert accelerator.use_seedable_sampler is False, "use_seedable_sampler should be False by default"
            assert "dispatch_batches" in deprecation_warning
            assert "split_batches" in deprecation_warning
            assert "even_batches" not in deprecation_warning
            assert "use_seedable_sampler" not in deprecation_warning

        # Pass in some arguments, but with their defaults
        with pytest.warns(FutureWarning) as cm:
            accelerator = Accelerator(
                even_batches=True,
                use_seedable_sampler=False,
            )
            deprecation_warning = str(cm.list[0].message)
            assert "even_batches" in deprecation_warning
            assert accelerator.even_batches is True
            assert "use_seedable_sampler" in deprecation_warning
            assert accelerator.use_seedable_sampler is False

    @require_non_cpu
    def test_accelerator_can_be_reinstantiated(self):
        _ = Accelerator()
        assert PartialState._shared_state["_cpu"] is False
        assert PartialState._shared_state["device"].type in ["cuda", "mps", "npu", "xpu", "xla"]
        with self.assertRaises(ValueError):
            _ = Accelerator(cpu=True)

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

    def test_free_memory_dereferences_prepared_components(self):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)
        accelerator.free_memory()

        assert len(accelerator._models) == 0
        assert len(accelerator._optimizers) == 0
        assert len(accelerator._schedulers) == 0
        assert len(accelerator._dataloaders) == 0

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

    @parameterized.expand((True, False), name_func=parameterized_custom_name_func)
    def test_save_load_model(self, use_safetensors):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
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
        assert (
            getattr(dummy_obj, "_is_accelerate_prepared", False) is False
        ), "Dummy object should have `_is_accelerate_prepared` set to `True`"
        assert (
            getattr(model, "_is_accelerate_prepared", False) is True
        ), "Model is missing `_is_accelerator_prepared` or is set to `False`"
        assert (
            getattr(optimizer, "_is_accelerate_prepared", False) is True
        ), "Optimizer is missing `_is_accelerator_prepared` or is set to `False`"
        assert (
            getattr(scheduler, "_is_accelerate_prepared", False) is True
        ), "Scheduler is missing `_is_accelerator_prepared` or is set to `False`"
        assert (
            getattr(train_dl, "_is_accelerate_prepared", False) is True
        ), "Train Dataloader is missing `_is_accelerator_prepared` or is set to `False`"
        assert (
            getattr(valid_dl, "_is_accelerate_prepared", False) is True
        ), "Valid Dataloader is missing `_is_accelerator_prepared` or is set to `False`"

    @require_non_torch_xla
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

    @require_non_torch_xla
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

        PartialState._reset_state()

    @require_non_torch_xla
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
