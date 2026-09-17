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

"""Offline causal-LM update parity for DDP, accumulation and mixed precision."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs, gather_object, set_seed


def flatten_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def create_model():
    """Create a tiny, randomly initialized Gemma 4 text model for offline tests."""
    return Gemma4ForCausalLM(
        Gemma4TextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            max_position_embeddings=512,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=16,
            num_kv_shared_layers=0,
            enable_moe_block=False,
            use_bidirectional_attention=None,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            use_cache=False,
            attn_implementation="eager",
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def train_reference(model, optimizer, input_ids, effective_batch_size, mixed_precision, device):
    losses, skipped, scales, interior_unchanged, weights_changed = [], [], [], [], []

    # Independent ordinary-PyTorch loop: one full effective batch per update,
    # with no Accelerator preparation, backward or optimizer wrapper.
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, enabled=mixed_precision == "fp16")
    for update_idx, batch in enumerate(input_ids.split(effective_batch_size)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype) if dtype else nullcontext():
            loss = model(input_ids=batch.to(device), labels=batch.to(device)).loss
        scaler.scale(loss).backward()

        # One controlled overflow checks skip and recovery through the integration.
        if mixed_precision == "fp16" and update_idx == 1:
            next(model.parameters()).grad.fill_(float("inf"))
        parameters_before_step = flatten_parameters(model).clone()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        did_skip = scaler.get_scale() < previous_scale
        if did_skip:
            assert torch.equal(parameters_before_step, flatten_parameters(model))
        weights_changed.append(not torch.equal(parameters_before_step, flatten_parameters(model)))
        losses.append(loss.detach().item())
        skipped.append(did_skip)
        scales.append(scaler.get_scale())

    return (
        model,
        losses,
        {
            "weights_changed": weights_changed,
            "skipped": skipped,
            "scales": scales,
            "interior_unchanged": interior_unchanged,
        },
    )


def train_accelerated(
    model, optimizer, input_ids, effective_batch_size, accelerator, mixed_precision, accumulation_steps
):
    losses, skipped, scales, interior_unchanged, weights_changed = [], [], [], [], []

    microbatch_size = effective_batch_size // (accelerator.num_processes * accumulation_steps)
    dataloader = DataLoader(TensorDataset(input_ids), batch_size=microbatch_size, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    window_loss = torch.zeros((), device=accelerator.device)
    for (batch,) in dataloader:
        with accelerator.accumulate(model):
            parameters_before_step = flatten_parameters(model).clone()
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)

            if mixed_precision == "fp16" and len(losses) == 1 and accelerator.sync_gradients:
                next(model.parameters()).grad.fill_(float("inf"))
            optimizer.step()
            optimizer.zero_grad()

            window_loss += loss.detach() / accumulation_steps
            if accelerator.sync_gradients:
                did_skip = accelerator.optimizer_step_was_skipped
                if did_skip:
                    assert torch.equal(parameters_before_step, flatten_parameters(model))
                # Every microbatch has equal valid-target counts, so this mean
                # reports the same effective-batch objective as the reference.
                losses.append(accelerator.reduce(window_loss, reduction="mean").item())
                weights_changed.append(not torch.equal(parameters_before_step, flatten_parameters(model)))
                skipped.append(did_skip)
                scales.append(accelerator.scaler.get_scale() if accelerator.scaler else 1.0)
                window_loss.zero_()
            else:
                interior_unchanged.append(torch.equal(parameters_before_step, flatten_parameters(model)))
    accelerator.wait_for_everyone()

    return (
        model,
        losses,
        {
            "weights_changed": weights_changed,
            "skipped": skipped,
            "scales": scales,
            "interior_unchanged": interior_unchanged,
        },
    )


def main():
    args = parse_args()

    accelerator = None
    if not args.reference:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[GradScalerKwargs(init_scale=128.0)],
        )
    device = torch.device("cuda:0") if args.reference else accelerator.device

    # Limit numerical variation when comparing different batch layouts and process counts.
    torch.set_float32_matmul_precision("highest")
    set_seed(1337, deterministic=True)

    num_updates = 10
    effective_batch_size = 8
    input_ids = torch.randint(
        3, 32, (num_updates * effective_batch_size, 12), generator=torch.Generator().manual_seed(1337)
    )

    model = create_model()
    initial_parameters = flatten_parameters(model).cpu().clone()
    if args.reference:
        model.to(device)
    model.train()

    compute_dtypes = set()
    # Observe a real operation inside forward: numerical tolerance alone could miss
    # a wrapper that silently failed to enable autocast.
    linear = next(module for module in model.modules() if isinstance(module, torch.nn.Linear))
    hook = linear.register_forward_hook(lambda module, inputs, output: compute_dtypes.add(str(output.dtype)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    if args.reference:
        model, losses, observations = train_reference(
            model, optimizer, input_ids, effective_batch_size, args.mixed_precision, device
        )
    else:
        model, losses, observations = train_accelerated(
            model,
            optimizer,
            input_ids,
            effective_batch_size,
            accelerator=accelerator,
            mixed_precision=args.mixed_precision,
            accumulation_steps=args.gradient_accumulation_steps,
        )

    hook.remove()
    observations["compute_dtypes"] = sorted(compute_dtypes)
    observations["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    observations = [observations]
    if accelerator:
        # Observe every rank, not just the writer. DDP replicas should be identical
        # even when different reduction/batch orders differ from the reference.
        replicas = accelerator.gather(flatten_parameters(model).unsqueeze(0))
        for replica in replicas[1:]:
            torch.testing.assert_close(replicas[0], replica, rtol=0, atol=0)
        observations = gather_object(observations)
    if args.reference or accelerator.is_main_process:
        args.output.write_text(
            json.dumps(
                {
                    "losses": losses,
                    "parameters": flatten_parameters(model).cpu().tolist(),
                    "update_norm": (flatten_parameters(model).cpu() - initial_parameters).norm().item(),
                    "world_size": 1 if args.reference else accelerator.num_processes,
                    "ranks": observations,
                },
                allow_nan=False,
            )
        )
    if accelerator:
        accelerator.end_training()


if __name__ == "__main__":
    main()
