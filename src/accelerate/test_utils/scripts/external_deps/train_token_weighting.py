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

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_constant_schedule, set_seed

from accelerate import Accelerator


def run_full_batch_reference(model, input_ids, labels, attention_mask, *, learning_rate):
    """Record one full-batch PyTorch update without changing the example's starting model."""
    initial_parameters = parameters_to_vector(model.parameters()).detach()

    reference_model = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=learning_rate)

    # Compute the full-batch mean independently so a scaling bug cannot affect both paths.
    logits = reference_model(input_ids=input_ids, attention_mask=attention_mask).logits
    reference_loss = F.cross_entropy(
        input=logits[:, :-1].reshape(-1, model.config.vocab_size),
        target=labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    reference_loss.backward()
    reference_gradients = torch.cat([parameter.grad.detach().flatten() for parameter in reference_model.parameters()])

    reference_optimizer.step()
    reference_parameters = parameters_to_vector(reference_model.parameters()).detach()

    return {
        "reference_gradients": reference_gradients,
        "reference_parameters": reference_parameters,
        "reference_update_norm": (reference_parameters - initial_parameters).norm().item(),
    }


def run_example(example_file, model, samples, *, learning_rate, batch_size, gradient_accumulation_steps):
    """Run the real example and record its target counts, gradients and updated parameters."""
    # Load the selected example by path so we can call its training function.
    spec = importlib.util.spec_from_file_location("autoregressive_example", example_file)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    target_counts = []
    gradients = []

    def record_target_count(module, args, kwargs):
        # Evaluation forwards must not count as training microbatches.
        if module.training:
            target_counts.append(kwargs["labels"][:, 1:].ne(-100).sum().item())

    def record_gradients(optimizer, args, kwargs):
        gradients.append(torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()]))

    target_count_hook = model.register_forward_pre_hook(record_target_count, with_kwargs=True)

    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Observe gradients before AdamW, whose adaptive update can hide a scaling error.
    optimizer.register_step_pre_hook(record_gradients)
    lr_scheduler = get_constant_schedule(optimizer)
    train_dataloader = DataLoader(samples, batch_size=batch_size)
    eval_dataloader = DataLoader(samples, batch_size=2)

    set_seed(1337)
    example.train(
        accelerator,
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        num_epochs=1,
        # Leave gradients unclipped, matching the full-batch reference.
        max_grad_norm=float("inf"),
    )
    accelerator.end_training()
    target_count_hook.remove()

    return {
        "target_counts": target_counts,
        "gradients": gradients,
        "parameters": parameters_to_vector(model.parameters()).detach(),
    }


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

    # Prepare inputs with unequal target counts to expose incorrect microbatch averaging.
    input_ids = torch.randint(1, model.config.vocab_size, (8, 8), generator=torch.Generator().manual_seed(1337))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    for row, length in enumerate([2, 3, 7, 8, 4, 5, 2, 6]):
        input_ids[row, length:] = 0
        labels[row, length:] = -100
        attention_mask[row, length:] = 0

    # Keep this token as context, but exclude its prediction from the loss.
    # Counting targets from the attention mask would incorrectly include it.
    labels[2, 3] = -100

    samples = [
        dict(input_ids=ids, labels=targets, attention_mask=mask)
        for ids, targets, mask in zip(input_ids, labels, attention_mask)
    ]

    learning_rate = 0.01
    reference = run_full_batch_reference(model, input_ids, labels, attention_mask, learning_rate=learning_rate)

    # Run the example and record its gradients and update.
    observations = run_example(
        args.example,
        model,
        samples,
        learning_rate=learning_rate,
        batch_size=2,
        gradient_accumulation_steps=2,
    )

    result = {**observations, **reference}
    output_file = args.output.with_suffix(f".rank{os.environ['RANK']}.pt")
    torch.save(result, output_file)


if __name__ == "__main__":
    main()
