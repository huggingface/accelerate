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

from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs, gather_object, set_seed


NUM_UPDATE_ATTEMPTS = 10
EFFECTIVE_BATCH_SIZE = 8
INITIAL_LOSS_SCALE = 128.0
LOSS_SCALE_BACKOFF = 0.5
# Keep scale growth outside this short experiment; only an injected fault lowers it.
SCALER_KWARGS = dict(init_scale=INITIAL_LOSS_SCALE, backoff_factor=LOSS_SCALE_BACKOFF, growth_interval=2000)


def flatten_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def create_model():
    """Create a tiny, randomly initialized Gemma 4 text model for offline tests."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--inject-nonfinite-at-attempt",
        type=int,
        default=None,
        help="Test-only fault: inject infinite gradients on every rank at this zero-based update attempt (FP16 only).",
    )
    args = parser.parse_args(argv)
    if args.gradient_accumulation_steps < 1:
        parser.error("gradient accumulation steps must be positive")
    if args.inject_nonfinite_at_attempt is not None:
        if args.mixed_precision != "fp16":
            parser.error("nonfinite-gradient injection requires FP16")
        if not 0 <= args.inject_nonfinite_at_attempt < NUM_UPDATE_ATTEMPTS:
            parser.error(f"injection attempt must be between 0 and {NUM_UPDATE_ATTEMPTS - 1}")
    return args


def microbatch_size_for(effective_batch_size, world_size, gradient_accumulation_steps):
    """Reject layouts that would silently truncate or change the effective batch."""
    if min(effective_batch_size, world_size, gradient_accumulation_steps) < 1:
        raise ValueError("batch size, world size and accumulation steps must be positive")
    microbatch_size, remainder = divmod(effective_batch_size, world_size * gradient_accumulation_steps)
    if remainder:
        raise ValueError("effective batch size must be divisible by world size * accumulation steps")
    return microbatch_size


def train_reference(
    model, optimizer, input_ids, effective_batch_size, mixed_precision, device, inject_nonfinite_at_attempt
):
    """Train full effective batches with ordinary PyTorch as the reference."""
    attempts = []

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16", **SCALER_KWARGS)

    for update_idx, batch in enumerate(input_ids.split(effective_batch_size)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype) if dtype else nullcontext():
            loss = model(input_ids=batch.to(device), labels=batch.to(device)).loss
        scaler.scale(loss).backward()

        # Faults are requested by the test, never implied by the precision mode.
        if update_idx == inject_nonfinite_at_attempt:
            next(model.parameters()).grad.fill_(float("inf"))
        parameters_before_step = flatten_parameters(model).clone()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        attempts.append(
            {
                "index": update_idx,
                "loss": loss.detach().item(),
                "parameters_changed": not torch.equal(parameters_before_step, flatten_parameters(model)),
                "step_was_skipped": scaler.get_scale() < previous_scale,
                "loss_scale": scaler.get_scale(),
            }
        )

    return model, {"attempts": attempts}


def train_with_accelerate(
    model,
    optimizer,
    input_ids,
    effective_batch_size,
    accelerator,
    gradient_accumulation_steps,
    inject_nonfinite_at_attempt,
):
    """Train distributed microbatches; let Accelerate own synchronization and scaling."""
    attempts, parameters_unchanged_during_accumulation = [], []
    update_attempt = 0

    microbatch_size = microbatch_size_for(effective_batch_size, accelerator.num_processes, gradient_accumulation_steps)
    dataloader = DataLoader(TensorDataset(input_ids), batch_size=microbatch_size, shuffle=False)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    window_loss = torch.zeros((), device=accelerator.device)

    for (batch,) in dataloader:
        with accelerator.accumulate(model):
            parameters_before_step = flatten_parameters(model).clone()
            loss = model(input_ids=batch, labels=batch).loss
            accelerator.backward(loss)

            # Inject after backward on every rank, only at the requested update boundary.
            # This does not test propagation of a fault originating on a single rank.
            if accelerator.sync_gradients and update_attempt == inject_nonfinite_at_attempt:
                next(model.parameters()).grad.fill_(float("inf"))
            optimizer.step()
            optimizer.zero_grad()

            window_loss += loss.detach() / gradient_accumulation_steps
            if accelerator.sync_gradients:
                # Every microbatch has equal valid-target counts, so this mean
                # reports the same effective-batch objective as the reference.
                attempts.append(
                    {
                        "index": update_attempt,
                        "loss": accelerator.reduce(window_loss, reduction="mean").item(),
                        "parameters_changed": not torch.equal(parameters_before_step, flatten_parameters(model)),
                        "step_was_skipped": accelerator.optimizer_step_was_skipped,
                        "loss_scale": accelerator.scaler.get_scale() if accelerator.scaler else 1.0,
                    }
                )
                update_attempt += 1
                window_loss.zero_()
            else:
                parameters_unchanged_during_accumulation.append(
                    torch.equal(parameters_before_step, flatten_parameters(model))
                )

    accelerator.wait_for_everyone()

    return model, {
        "attempts": attempts,
        "parameters_unchanged_during_accumulation": parameters_unchanged_during_accumulation,
    }


def main():
    args = parse_args()

    accelerator = None
    if not args.reference:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[GradScalerKwargs(**SCALER_KWARGS)],
        )
    microbatch_size_for(
        EFFECTIVE_BATCH_SIZE, 1 if args.reference else accelerator.num_processes, args.gradient_accumulation_steps
    )
    device = torch.device("cuda:0") if args.reference else accelerator.device

    # Limit numerical variation when comparing different batch layouts and process counts.
    torch.set_float32_matmul_precision("highest")
    set_seed(1337, deterministic=True)

    # Complete effective batches, with equal-length, unmasked targets in every row.
    # The mean-of-means comparison below does not cover unequal valid-token counts.
    input_ids = torch.randint(
        3, 32, (NUM_UPDATE_ATTEMPTS * EFFECTIVE_BATCH_SIZE, 12), generator=torch.Generator().manual_seed(1337)
    )

    model = create_model()
    initial_parameters = flatten_parameters(model).cpu().clone()
    if args.reference:
        model.to(device)
    model.train()

    linear_output_dtypes = set()
    # Observe a real operation inside forward: numerical tolerance alone could miss
    # a wrapper that silently failed to enable autocast.
    linear = next(module for module in model.modules() if isinstance(module, torch.nn.Linear))
    hook = linear.register_forward_hook(lambda module, inputs, output: linear_output_dtypes.add(str(output.dtype)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    if args.reference:
        model, observations = train_reference(
            model,
            optimizer,
            input_ids,
            EFFECTIVE_BATCH_SIZE,
            args.mixed_precision,
            device,
            args.inject_nonfinite_at_attempt,
        )
    else:
        model, observations = train_with_accelerate(
            model,
            optimizer,
            input_ids,
            EFFECTIVE_BATCH_SIZE,
            accelerator=accelerator,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            inject_nonfinite_at_attempt=args.inject_nonfinite_at_attempt,
        )

    hook.remove()
    observations["linear_output_dtypes"] = sorted(linear_output_dtypes)
    observations["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    observations = [observations]
    if accelerator:
        # DDP replicas must agree exactly, independently of reference tolerances.
        replicas = accelerator.gather(flatten_parameters(model).unsqueeze(0))
        for replica in replicas[1:]:
            torch.testing.assert_close(replicas[0], replica, rtol=0, atol=0)
        observations = gather_object(observations)

    if args.reference or accelerator.is_main_process:
        args.output.write_text(
            json.dumps(
                {
                    "parameters": flatten_parameters(model).cpu().tolist(),
                    "parameter_delta_norm": (flatten_parameters(model).cpu() - initial_parameters).norm().item(),
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
