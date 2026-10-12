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

"""Measure the autoregressive example's token-weighted gradients against a full-batch reference."""

import argparse
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Use FP32 to isolate normalization from reduced-precision rounding.
    model = AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.float32, use_cache=False
    )
    model.train()

    input_ids = torch.randint(1, model.config.vocab_size, (8, 8), generator=torch.Generator().manual_seed(1337))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    for row, length in enumerate([2, 3, 7, 8, 4, 5, 2, 6]):
        input_ids[row, length:] = 0
        labels[row, length:] = -100
        attention_mask[row, length:] = 0
    labels[2, 3] = -100  # Also ignore one non-padding target.

    samples = [
        dict(input_ids=ids, labels=targets, attention_mask=mask)
        for ids, targets, mask in zip(input_ids, labels, attention_mask)
    ]

    def get_dataloaders(accelerator, batch_size):
        return DataLoader(samples, batch_size=batch_size), DataLoader(samples, batch_size=2)

    initial_parameters = parameters_to_vector(model.parameters()).detach()

    # Independent oracle: one ordinary PyTorch update on the full batch. Compute
    # the shifted mean loss explicitly, not through the example's token scaling.
    reference_model = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.01)
    logits = reference_model(input_ids=input_ids, attention_mask=attention_mask).logits
    reference_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, model.config.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    reference_loss.backward()
    reference_gradients = torch.cat([parameter.grad.detach().flatten() for parameter in reference_model.parameters()])
    reference_optimizer.step()
    reference_parameters = parameters_to_vector(reference_model.parameters()).detach()

    spec = importlib.util.spec_from_file_location("autoregressive_example", args.example)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    target_counts = []
    gradients = []

    def observe_batch(module, args, kwargs):
        if module.training:
            target_counts.append(kwargs["labels"][:, 1:].ne(-100).sum().item())

    def observe_gradients(optimizer, args, kwargs):
        gradients.append(torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()]))

    target_count_hook = model.register_forward_pre_hook(observe_batch, with_kwargs=True)

    original_adamw = example.AdamW

    def make_optimizer(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        optimizer.register_step_pre_hook(observe_gradients)
        return optimizer

    # Leave gradients unclipped, matching the full-batch reference.
    config = {"lr": 0.01, "num_epochs": 1, "seed": 1337, "batch_size": 2, "max_grad_norm": float("inf")}
    training_args = SimpleNamespace(
        cpu=True, mixed_precision="no", gradient_accumulation_steps=2, with_wandb_tracking=False
    )
    # Keep the actual example's prepare/gather/no_sync/backward/optimizer loop.
    # Controlled target counts make normalization errors measurable.
    # Observe gradients before AdamW, whose adaptive update can hide a scaling error.
    with (
        patch.object(example, "get_dataloaders", get_dataloaders),
        patch.object(example.AutoModelForCausalLM, "from_pretrained", return_value=model),
        patch.object(example, "AdamW", make_optimizer),
    ):
        example.training_function(config, training_args)
    target_count_hook.remove()

    result = {
        "target_counts": target_counts,
        "gradients": gradients,
        "reference_gradients": reference_gradients,
        "parameters": parameters_to_vector(model.parameters()).detach(),
        "reference_parameters": reference_parameters,
        "reference_update_norm": (reference_parameters - initial_parameters).norm().item(),
    }
    output_file = args.output.with_suffix(f".rank{os.environ['RANK']}.pt")
    torch.save(result, output_file)


if __name__ == "__main__":
    main()
