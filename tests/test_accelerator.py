import json
import os
import pickle
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import DistributedType, infer_auto_device_map, init_empty_weights
from accelerate.accelerator import Accelerator
from accelerate.state import GradientState, PartialState
from accelerate.test_utils import require_bnb, require_multi_gpu, slow
from accelerate.test_utils.testing import AccelerateTestCase, require_cuda
from accelerate.utils import patch_environment


def create_components():
    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    train_dl = DataLoader(TensorDataset(torch.tensor([1, 2, 3])))
    valid_dl = DataLoader(TensorDataset(torch.tensor([4, 5, 6])))

    return model, optimizer, scheduler, train_dl, valid_dl


def get_signature(model):
    return (model.weight.abs().sum() + model.bias.abs().sum()).item()


def load_random_weights(model):
    state = torch.nn.Linear(*tuple(model.weight.T.shape)).state_dict()
    model.load_state_dict(state)


class AcceleratorTester(AccelerateTestCase):
    @require_cuda
    def test_accelerator_can_be_reinstantiated(self):
        _ = Accelerator()
        assert PartialState._shared_state["_cpu"] is False
        assert PartialState._shared_state["device"].type == "cuda"
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

        self.assertTrue(prepared_model in accelerator._models)
        self.assertTrue(prepared_optimizer in accelerator._optimizers)
        self.assertTrue(prepared_scheduler in accelerator._schedulers)
        self.assertTrue(prepared_train_dl in accelerator._dataloaders)
        self.assertTrue(prepared_valid_dl in accelerator._dataloaders)

    def test_free_memory_dereferences_prepared_components(self):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)
        accelerator.free_memory()

        self.assertTrue(len(accelerator._models) == 0)
        self.assertTrue(len(accelerator._optimizers) == 0)
        self.assertTrue(len(accelerator._schedulers) == 0)
        self.assertTrue(len(accelerator._dataloaders) == 0)

    def test_env_var_device(self):
        """Tests that setting the torch device with ACCELERATE_TORCH_DEVICE overrides default device."""
        PartialState._reset_state()

        # Mock torch.cuda.set_device to avoid an exception as the device doesn't exist
        def noop(*args, **kwargs):
            pass

        with patch("torch.cuda.set_device", noop), patch_environment(ACCELERATE_TORCH_DEVICE="cuda:64"):
            accelerator = Accelerator()
            self.assertEqual(str(accelerator.state.device), "cuda:64")

    def test_save_load_model(self):
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

        model_signature = get_signature(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname)

            # make sure random weights don't match
            load_random_weights(model)
            self.assertTrue(abs(model_signature - get_signature(model)) > 1e-3)

            # make sure loaded weights match
            accelerator.load_state(tmpdirname)
            self.assertTrue(abs(model_signature - get_signature(model)) < 1e-3)

    def test_save_load_model_with_hooks(self):
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
            with open(os.path.join(input_dir, "data.json"), "r") as f:
                config = json.load(f)

            models[0].class_name = config["class_name"]

        save_hook = accelerator.register_save_state_pre_hook(save_config)
        load_hook = accelerator.register_load_state_pre_hook(load_config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname)

            # make sure random weights don't match with hooks
            load_random_weights(model)
            self.assertTrue(abs(model_signature - get_signature(model)) > 1e-3)

            # random class name to verify correct one is loaded
            model.class_name = "random"

            # make sure loaded weights match with hooks
            accelerator.load_state(tmpdirname)
            self.assertTrue(abs(model_signature - get_signature(model)) < 1e-3)

            # mode.class_name is loaded from config
            self.assertTrue(model.class_name == model.__class__.__name__)

        # remove hooks
        save_hook.remove()
        load_hook.remove()

        with tempfile.TemporaryDirectory() as tmpdirname:
            accelerator.save_state(tmpdirname)

            # make sure random weights don't match with hooks removed
            load_random_weights(model)
            self.assertTrue(abs(model_signature - get_signature(model)) > 1e-3)

            # random class name to verify correct one is loaded
            model.class_name = "random"

            # make sure loaded weights match with hooks removed
            accelerator.load_state(tmpdirname)
            self.assertTrue(abs(model_signature - get_signature(model)) < 1e-3)

            # mode.class_name is NOT loaded from config
            self.assertTrue(model.class_name != model.__class__.__name__)

    def test_accelerator_none(self):
        """Just test that passing None to accelerator.prepare() works."""
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        dummy_obj = None

        # This should work
        model, optimizer, scheduler, train_dl, valid_dl, dummy_obj = accelerator.prepare(
            model, optimizer, scheduler, train_dl, valid_dl, dummy_obj
        )
        self.assertTrue(dummy_obj is None)

    def test_is_accelerator_prepared(self):
        """Checks that `_is_accelerator_prepared` is set properly"""
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dl, valid_dl = create_components()
        dummy_obj = [1, 2, 3]

        # This should work
        model, optimizer, scheduler, train_dl, valid_dl, dummy_obj = accelerator.prepare(
            model, optimizer, scheduler, train_dl, valid_dl, dummy_obj
        )
        self.assertEqual(
            getattr(dummy_obj, "_is_accelerate_prepared", False),
            False,
            "Dummy object should have `_is_accelerate_prepared` set to `True`",
        )
        self.assertEqual(
            getattr(model, "_is_accelerate_prepared", False),
            True,
            "Model is missing `_is_accelerator_prepared` or is set to `False`",
        )
        self.assertEqual(
            getattr(optimizer, "_is_accelerate_prepared", False),
            True,
            "Optimizer is missing `_is_accelerator_prepared` or is set to `False`",
        )
        self.assertEqual(
            getattr(scheduler, "_is_accelerate_prepared", False),
            True,
            "Scheduler is missing `_is_accelerator_prepared` or is set to `False`",
        )
        self.assertEqual(
            getattr(train_dl, "_is_accelerate_prepared", False),
            True,
            "Train Dataloader is missing `_is_accelerator_prepared` or is set to `False`",
        )
        self.assertEqual(
            getattr(valid_dl, "_is_accelerate_prepared", False),
            True,
            "Valid Dataloader is missing `_is_accelerator_prepared` or is set to `False`",
        )

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

    @slow
    @require_bnb
    @require_multi_gpu
    def test_accelerator_bnb_multi_gpu(self):
        """Tests that the accelerator can be used with the BNB library."""
        from transformers import AutoModelForCausalLM

        PartialState._shared_state = {"distributed_type": DistributedType.MULTI_GPU}

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

    @slow
    @require_bnb
    @require_multi_gpu
    def test_accelerator_bnb_multi_gpu_no_distributed(self):
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

    @require_cuda
    def test_accelerator_cpu_flag_prepare(self):
        model = torch.nn.Linear(10, 10)
        sgd = torch.optim.SGD(model.parameters(), lr=0.01)
        accelerator = Accelerator(cpu=True)
        _ = accelerator.prepare(sgd)

    @require_cuda
    def test_can_unwrap_model_fp16(self):
        # test for a regression introduced in #872
        # before the fix, after unwrapping with keep_fp32_wrapper=False, there would be the following error:
        # Linear.forward() missing 1 required positional argument: 'input'
        model = create_components()[0]
        accelerator = Accelerator(mixed_precision="fp16")
        inputs = torch.randn(10, 2).cuda()
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
