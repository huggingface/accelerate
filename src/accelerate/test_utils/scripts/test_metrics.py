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
    dset = RegressionDataset(length=80)
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
    for batch in dataloader:
        ddp_input, ddp_target = batch.values()
        # First do single process
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        with torch.no_grad():
            logits = model(input)
            accuracy_single = accuracy(logits.argmax(dim=-1), target)
        # Then do multiprocess
        with torch.no_grad():
            logits = ddp_model(ddp_input)
            logits, target = accelerator.gather_for_metrics((logits, ddp_target), dataloader)
            accuracy_multi = accuracy(logits.argmax(dim=-1), target)
        assert torch.allclose(accuracy_single, accuracy_multi), "The two accuracies were not the same!"


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
