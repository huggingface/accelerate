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


def get_setup(accelerator):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=82)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    return model, ddp_model, dataloader


def accuracy(predictions, labels) -> float:
    """
    Get the accuracy with respect to the most likely label
    """
    return (predictions == labels).float().mean()


def test_torch_metrics():
    accelerator = Accelerator()
    model, ddp_model, dataloader = get_setup(accelerator)
    single_device_logs_and_targs = []
    multi_device_logs_and_targs = []
    for batch in dataloader:
        ddp_input, ddp_target = batch.values()
        with torch.no_grad():
            logits = ddp_model(ddp_input)
            logits, target = accelerator.gather_for_metrics((logits, ddp_target), dataloader)
            multi_device_logs_and_targs.append((logits, target))
    inps, targs = [], []
    for (inp, targ) in multi_device_logs_and_targs:
        inps.append(inp)
        targs.append(targ)
    inps, targs = torch.cat(inps), torch.cat(targs)
    # inps, targs = inps[:dataloader.total_dataset_length], targs[:dataloader.total_dataset_length]
    assert len(inps) == 82, f'seen_logits: {len(inps)}\ntargs: \n{targs}'


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.local_process_index == 0:
        print("**Test torch metrics**")
    test_torch_metrics()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
