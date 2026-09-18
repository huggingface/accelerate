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

"""Check the autoregressive accumulation example against a full-batch update."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM


def flatten_parameters(model):
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=11,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            attention_dropout=0.0,
            pad_token_id=0,
            use_cache=False,
            attn_implementation="eager",
        )
    )
    input_ids = torch.randint(1, 11, (8, 8), generator=torch.Generator().manual_seed(17))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    for row, length in enumerate([2, 3, 7, 8, 4, 5, 2, 6]):
        input_ids[row, length:] = 0
        labels[row, length:] = -100
        attention_mask[row, length:] = 0
    labels[2, 3] = -100  # Also ignore one non-padding target.
    initial_parameters = flatten_parameters(model).clone()

    # Independent oracle: one ordinary PyTorch update on the full batch. Compute
    # the shifted mean loss explicitly, not through the example's token scaling.
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    logits = reference(input_ids=input_ids, attention_mask=attention_mask).logits
    reference_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
    reference_loss.backward()
    reference_gradients = torch.cat([parameter.grad.detach().flatten() for parameter in reference.parameters()])
    reference_optimizer.step()
    reference_parameters = flatten_parameters(reference)

    records = []

    def observe_batch(module, positional, kwargs):
        if module.training:
            records.append(
                {
                    "tokens": kwargs["labels"][:, 1:].ne(-100).sum().item(),
                    "denominator": kwargs.get("num_items_in_batch"),
                }
            )

    hook = model.register_forward_pre_hook(observe_batch, with_kwargs=True)
    samples = [
        dict(input_ids=ids, labels=targets, attention_mask=mask)
        for ids, targets, mask in zip(input_ids, labels, attention_mask)
    ]

    def get_dataloaders(accelerator, batch_size):
        return DataLoader(samples, batch_size=batch_size), DataLoader(samples, batch_size=2)

    spec = importlib.util.spec_from_file_location("autoregressive_example", args.example)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    gradients = []
    adamw = example.AdamW

    def observe_gradients(optimizer, args, kwargs):
        gradients.append(torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()]))

    def make_optimizer(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        optimizer.register_step_pre_hook(observe_gradients)
        return optimizer

    config = {"lr": 0.01, "num_epochs": 1, "seed": 42, "batch_size": 2, "max_grad_norm": float("inf")}
    training_args = SimpleNamespace(
        cpu=True, mixed_precision="no", gradient_accumulation_steps=2, with_wandb_tracking=False
    )
    # Keep the actual example's prepare/gather/no_sync/backward/optimizer loop.
    # Use local model/data loading and observe the real optimizer before it steps.
    with (
        patch.object(example, "get_dataloaders", get_dataloaders),
        patch.object(example.AutoModelForCausalLM, "from_pretrained", return_value=model),
        patch.object(example, "AdamW", make_optimizer),
    ):
        example.training_function(config, training_args)
    hook.remove()

    args.output.with_suffix(f".rank{os.environ['RANK']}.json").write_text(
        json.dumps(
            {
                "batches": records,
                "gradients": [gradient.tolist() for gradient in gradients],
                "reference_gradients": reference_gradients.tolist(),
                "parameters": flatten_parameters(model).tolist(),
                "reference_parameters": reference_parameters.tolist(),
                "reference_update_norm": (reference_parameters - initial_parameters).norm().item(),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
