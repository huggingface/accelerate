# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Checks that `save_fsdp_model`/`load_fsdp_model` round-trip a PEFT adapter without losing any rank's shard.

Every rank fills its *local* shard of every adapter parameter with a rank-distinctive constant, so that a shard that
was gathered from the wrong rank (or never restored at all) is unambiguous. The freshly built model is zeroed before
loading, otherwise a value that was simply never overwritten would look like a successful load.
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.tensor import DTensor

from accelerate import Accelerator
from accelerate.utils import load_fsdp_model, save_fsdp_model


DIM = 64


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


class TinyModel(nn.Module):
    def __init__(self, dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def build(accelerator):
    model = get_peft_model(TinyModel(DIM), LoraConfig(r=8, lora_alpha=16, target_modules=["lin1", "lin2"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    # FSDP2 requires the model and the optimizer to be prepared together
    return accelerator.prepare(model, optimizer)


def adapter_parameters(model):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, param


def local(param):
    return param.to_local() if isinstance(param.data, DTensor) else param.data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--adapter_only", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    fsdp_plugin = accelerator.state.fsdp_plugin
    kwargs = {"adapter_only": True} if args.adapter_only else {}
    # a value no rank shares with another, so that a mixed-up shard is detectable
    fill_value = 1.0 + accelerator.process_index

    model, optimizer = build(accelerator)
    with torch.no_grad():
        for _, param in adapter_parameters(model):
            local(param).fill_(fill_value)

    save_fsdp_model(fsdp_plugin, accelerator, model, args.output_dir, **kwargs)
    accelerator.wait_for_everyone()

    del model, optimizer
    model, optimizer = build(accelerator)
    with torch.no_grad():
        for _, param in adapter_parameters(model):
            local(param).zero_()

    load_fsdp_model(fsdp_plugin, accelerator, model, args.output_dir, **kwargs)
    accelerator.wait_for_everyone()

    num_checked = 0
    for name, param in adapter_parameters(model):
        shard = local(param)
        if shard.numel() == 0:
            # this rank holds no slice of this parameter
            continue
        num_checked += 1
        assert torch.allclose(shard, torch.full_like(shard, fill_value)), (
            f"rank {accelerator.process_index} did not get its shard of {name} back: expected {fill_value}, "
            f"got values in [{shard.min().item()}, {shard.max().item()}]"
        )
    assert num_checked > 0, f"rank {accelerator.process_index} checked no adapter parameter"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"**PEFT {'adapter-only ' if args.adapter_only else ''}checkpointing round-trip is correct**")


if __name__ == "__main__":
    main()
