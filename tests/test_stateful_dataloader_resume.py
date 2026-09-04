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

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate.data_loader import prepare_data_loader
from accelerate.test_utils.testing import require_torchdata_stateful_dataloader


@require_torchdata_stateful_dataloader
def test_distributed_stateful_dataloader_restores_shuffle_order():
    def make_loader(process_index):
        torch.manual_seed(1234)
        return prepare_data_loader(
            DataLoader(TensorDataset(torch.arange(64)), batch_size=4, shuffle=True, num_workers=0),
            num_processes=2,
            process_index=process_index,
            use_stateful_dataloader=True,
        )

    for process_index in (0, 1):
        loader = make_loader(process_index)
        list(loader)

        iterator = iter(loader)
        for _ in range(3):
            next(iterator)
        state = loader.state_dict()
        expected = [batch[0].tolist() for batch in iterator]

        resumed_loader = make_loader(process_index)
        resumed_loader.load_state_dict(state)
        resumed = [batch[0].tolist() for batch in resumed_loader]
        assert resumed == expected

        resumed_loader = make_loader(process_index)
        resumed_loader.load_state_dict(state)
        resumed_iterator = iter(resumed_loader)
        next(resumed_iterator)
        second_state = resumed_loader.state_dict()
        second_expected = [batch[0].tolist() for batch in resumed_iterator]

        second_resumed_loader = make_loader(process_index)
        second_resumed_loader.load_state_dict(second_state)
        second_resumed = [batch[0].tolist() for batch in second_resumed_loader]
        assert second_resumed == second_expected
