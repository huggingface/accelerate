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

"""Offline FP32 causal-LM training worker for single-process versus DDP parity."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

from accelerate import Accelerator
from accelerate.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    accelerator = Accelerator(mixed_precision="no")
    set_seed(1234)

    steps, global_batch = 10, 8
    # Separate data and model RNGs keep initialization and samples identical.
    tokens = torch.randint(0, 32, (steps * global_batch, 12), generator=torch.Generator().manual_seed(1234))
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=0,
            eos_token_id=1,
            use_cache=False,
            attn_implementation="eager",
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # prepare shards whole batches. Two ranks each consume four rows per update;
    # the single-process reference consumes the same eight rows together.
    loader = DataLoader(TensorDataset(tokens), batch_size=global_batch // accelerator.num_processes, shuffle=False)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    losses = []
    for (input_ids,) in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        accelerator.backward(loss)
        optimizer.step()
        # All local batches are full and each row has eleven predicted tokens,
        # so averaging rank losses gives the global token-mean loss.
        losses.append(accelerator.reduce(loss.detach(), reduction="mean").item())

    if accelerator.is_main_process:
        parameters = torch.cat([p.detach().flatten().cpu() for p in accelerator.unwrap_model(model).parameters()])
        args.output.write_text(
            json.dumps(
                {"losses": losses, "parameters": parameters.tolist(), "world_size": accelerator.num_processes},
                allow_nan=False,
            )
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
