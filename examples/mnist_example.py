# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a fully convolutional model on MNIST data
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
########################################################################


from __future__ import print_function

import argparse
import os
import os.path
import threading
from functools import partial
from typing import Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm as original_tqdm

from accelerate import Accelerator

class Accuracy:
    """Accuracy score.

    Notes
    -----
    This `Accuracy` class is used to compute the accuracy epoch-wise. Although
    accuracy can easily be computed without resorting to such a class, the
    same logic may be used to compute other metrics epoch-wise (e.g.
    ROC-AUC, PR-AUC scores)."""
    def __init__(self):
        super().__init__()
        self.__build()

    def __build(self):
        self._lock = threading.Lock()
        self._predictions = []
        self._targets = []

    def reset(self):
        self._predictions.clear()
        self._targets.clear()

    def update(self, output):
        y_pred, y_true = output
        with self._lock:
            self._predictions.append(y_pred)
            self._targets.append(y_true)

    def compute(self):
        with self._lock:
            predictions = torch.cat(self._predictions, dim=0).numpy()
            targets = torch.cat(self._targets, dim=0).numpy()
            print(f'Shapes: predictions {predictions.shape}, targets {targets.shape}')
            return accuracy_score(y_true=targets, y_pred=predictions)  # or any other metric computed epoch-wise


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--per_device_eval_batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='The per-device batch size to use for evaluation.')
    parser.add_argument('--per_device_train_batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='The per-device batch size to use for training.')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        metavar='N',
                        help='Number of epochs to train the model (default: 5)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='Learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--out_dir',
                        type=str,
                        help='Path where the trained model will be saved (if not None).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    accelerator = Accelerator()
    tqdm = partial(original_tqdm, disable=not accelerator.is_main_process, position=0)

    # TRAIN AND TEST DATASETS/DATALOADERS
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Download MNIST only on main process
    with accelerator.main_process_first():

        train_dataset = datasets.MNIST(os.environ['DSDIR'],
                                       train=True,
                                       download=True,
                                       transform=train_transforms)

        test_dataset = datasets.MNIST(os.environ['DSDIR'],
                                      train=False,
                                      download=True,
                                      transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.per_device_train_batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.per_device_eval_batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    model = Net().to(accelerator.device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    def evaluate(_model: nn.Module,
                 _device: Union[torch.device, str],
                 _test_loader: DataLoader,
                 _epoch: int):
        _model.eval()

        test_accuracy = Accuracy()

        with torch.no_grad():
            for data, target in tqdm(_test_loader, desc=f'Eval (epoch {_epoch:03d})'):

                data, target = data.to(_device, non_blocking=True), target.to(_device, non_blocking=True)
                output = _model(data)
                preds = output.argmax(dim=1, keepdim=True)

                # Compute accuracy:
                # We first gather predictions & labels from all processes
                bs = _test_loader.batch_size
                y_pred = accelerator.gather(preds).detach().cpu()[:bs]
                y_true = accelerator.gather(target).detach().cpu()[:bs]
                test_accuracy.update((y_pred, y_true))

        test_acc = test_accuracy.compute()
        test_accuracy.reset()
        return test_acc

    def train_one_epoch(_args: Dict[str, Any],
                        _model: nn.Module,
                        _device: Union[torch.device, str],
                        _train_loader: DataLoader,
                        _optimizer: Optimizer,
                        _epoch: int):
        _model.train()
        for batch in tqdm(_train_loader, desc=f'train (epoch {_epoch:03d})'):
            data, target = batch
            data, target = data.to(_device), target.to(_device)
            _optimizer.zero_grad()
            output = _model(data)
            loss = F.nll_loss(output, target)
            accelerator.backward(loss)
            _optimizer.step()

    # TRAINING
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, accelerator.device, train_dataloader, optimizer, epoch)
        eval_accuracy = evaluate(model, accelerator.device, test_dataloader, epoch)
        if accelerator.is_main_process:
            print(f'Epoch {epoch:02d} / Eval accuracy = {eval_accuracy}')
        scheduler.step()

    # SAVE TRAINED MODEL (OPTIONAL)
    if accelerator.is_main_process and args.out_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(args.out_dir, save_function=accelerator.save)


if __name__ == '__main__':

    main()
