import json
import os
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate.accelerator import Accelerator
from accelerate.state import PartialState
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
        _ = Accelerator(cpu=True)
        assert PartialState._shared_state["_cpu"] is True
        assert PartialState._shared_state["device"].type == "cpu"

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
