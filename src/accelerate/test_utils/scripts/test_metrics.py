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

from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import set_seed


def get_setup(accelerator, num_samples=82):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=num_samples)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    return model, ddp_model, dataloader


def generate_predictions(model, dataloader, accelerator):
    logits_and_targets = []
    for batch in dataloader:
        input, target = batch.values()
        with torch.no_grad():
            logits = model(input)
            logits, target = accelerator.gather_for_metrics((logits, target))
            logits_and_targets.append((logits, target))
    inps, targs = [], []
    for (inp, targ) in logits_and_targets:
        inps.append(inp)
        targs.append(targ)
    inps, targs = torch.cat(inps), torch.cat(targs)
    return inps, targs


def test_torch_metrics(accelerator: Accelerator, num_samples=82):
    model, ddp_model, dataloader = get_setup(accelerator, num_samples)
    inps, targs = generate_predictions(ddp_model, dataloader, accelerator)
    assert (
        len(inps) == num_samples
    ), f"Unexpected number of inputs:\n    Expected: {num_samples}\n    Actual: {len(inps)}"


def main():
    accelerator = Accelerator(split_batches=False, dispatch_batches=False)
    if accelerator.is_local_main_process:
        print("**Test torch metrics**")
        print("With: `split_batches=False`, `dispatch_batches=False`")
    test_torch_metrics(accelerator)
    accelerator.state._reset_state()
    accelerator = Accelerator(split_batches=True, dispatch_batches=False)
    if accelerator.is_local_main_process:
        print("With: `split_batches=True`, `dispatch_batches=False`")
    test_torch_metrics(accelerator)
    accelerator.state._reset_state()
    accelerator = Accelerator(split_batches=False, dispatch_batches=True)
    if accelerator.is_local_main_process:
        print("With: `split_batches=False`, `dispatch_batches=True`")
    test_torch_metrics(accelerator)
    accelerator.state._reset_state()
    accelerator = Accelerator(split_batches=True, dispatch_batches=True)
    if accelerator.is_local_main_process:
        print("With: `split_batches=True`, `dispatch_batches=True`")
    test_torch_metrics(accelerator)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
