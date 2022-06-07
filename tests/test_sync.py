# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import torch.nn.functional as F
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel, require_multi_gpu, require_cuda, require_cpu
from accelerate.utils import set_seed


class SyncTester(unittest.TestCase):
    @staticmethod
    def step_model(accelerator, model, input, target):
        model.train()
        output = model(input.to(accelerator.device))
        loss = F.mse_loss(output, target.to(accelerator.device))
        accelerator.backward(loss)

    @staticmethod
    def setup_training():
        modelA = RegressionModel()
        modelB = RegressionModel()
        dataset = RegressionDataset(length=6)
        dataloader = DataLoader(dataset, batch_size=2)
        return modelA, modelB, dataloader

    @require_cpu
    def test_ignored_accumulation(self):
        accelerator = Accelerator()
        set_seed(42)
        modelA, modelB, dataloader = self.setup_training()
        dataloader, modelA, modelB = accelerator.prepare(dataloader, modelA, modelB)
        modelA.to(accelerator.device)
        modelB.to(accelerator.device)
        # Check two model parameters over three batches
        for iteration, batch in enumerate(dataloader):
            x,y = batch["x"], batch["y"]
            self.step_model(accelerator, modelA, x, y)
            if iteration % 2 == 0:
                # Accumulate locally
                with accelerator.no_sync(modelB):
                    self.step_model(accelerator, modelB, x, y)
            else:
                # Sync
                self.step_model(accelerator, modelB, x, y)

            # Make sure they align
            for i, j in zip(modelA.parameters(), modelB.parameters()):
                if not i.requires_grad:
                    continue
                self.assertEqual(i.grad, j.grad)


    @require_multi_gpu
    def test_gradient_accumulation(self):
        accelerator = Accelerator()
        set_seed(42)
        modelA, modelB, dataloader = self.setup_training()
        dataloader, modelA, modelB = accelerator.prepare(dataloader, modelA, modelB)
        modelA.to(accelerator.device)
        modelB.to(accelerator.device)
        # Check two model parameters over three batches
        for iteration, batch in enumerate(dataloader):
            x,y = batch["x"], batch["y"]
            self.step_model(accelerator, modelA, x, y)
            if iteration % 2 == 0:
                # Accumulate locally
                with accelerator.no_sync(modelB):
                    self.step_model(accelerator, modelB, x, y)
            else:
                # Sync
                self.step_model(accelerator, modelB, x, y)

            # Make sure they align
            for i, j in zip(modelA.parameters(), modelB.parameters()):
                if not i.requires_grad:
                    continue
                if iteration % 2 == 0:
                    self.assertNotEqual(i.grad, j.grad)
                else:
                    self.assertEqual(i.grad, j.grad)
