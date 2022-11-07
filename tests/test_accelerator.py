import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate.accelerator import Accelerator


def create_components():
    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    train_dl = DataLoader(TensorDataset(torch.tensor([1, 2, 3])))
    valid_dl = DataLoader(TensorDataset(torch.tensor([4, 5, 6])))

    return model, optimizer, scheduler, train_dl, valid_dl


class AcceleratorTester(unittest.TestCase):
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
