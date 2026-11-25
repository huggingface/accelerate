# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from torch.distributed.fsdp import FSDPModule

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin


def main():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_cls_names_to_wrap=["Linear"],
        activation_checkpointing=True,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    shared = torch.nn.Linear(4, 4)
    # Checkpointing wraps `model[1]` only, so calls through `model[2]` bypass the wrapper's FSDP hooks.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), shared, shared)
    model, _ = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))

    assert isinstance(model[0], FSDPModule) and not isinstance(model[0]._checkpoint_wrapped_module, FSDPModule)
    assert isinstance(model[2], FSDPModule)

    accelerator.end_training()


if __name__ == "__main__":
    main()
