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
"""Raw Accelerator training loop over a tiny causal LM and hermetic synthetic tokens.

Backend (DDP, FSDP, DeepSpeed) is selected by the launcher config passed to
``accelerate launch --config_file``; this script itself is backend-agnostic.

The main process writes a JSON summary (loss history, step count, distributed_type,
mixed_precision) to ``--output_file`` for the caller to assert on.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_file", type=Path, required=True)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Directory for save_state()/load_state(). Required when --save_at_step or --resume_from is set.",
    )
    p.add_argument(
        "--save_at_step",
        type=int,
        default=None,
        help="After this many steps, call save_state() into --checkpoint_dir and exit.",
    )
    p.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Call load_state() from this path before training begins.",
    )
    p.add_argument("--num_sequences", type=int, default=32, help="Size of the memorizable synthetic corpus.")
    p.add_argument("--sequence_length", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--num_attention_heads", type=int, default=2)
    return p.parse_args()


def build_model(args):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        max_position_embeddings=args.sequence_length,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def build_dataset(args):
    # REASON: a small fixed corpus repeated across epochs is intentionally memorizable,
    # so downstream convergence assertions have a reliable signal.
    gen = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(0, args.vocab_size, (args.num_sequences, args.sequence_length), generator=gen)
    return TensorDataset(tokens, tokens.clone())


def main():
    args = parse_args()

    set_seed(args.seed, deterministic=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    model = build_model(args)
    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if args.resume_from is not None:
        accelerator.load_state(str(args.resume_from))

    loss_history = []
    step = 0
    saved_and_exiting = False

    while step < args.max_steps and not saved_and_exiting:
        for input_ids, labels in dataloader:
            if step >= args.max_steps:
                break
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            gathered = accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
            loss_history.append(gathered.mean().item())
            step += 1

            if args.save_at_step is not None and step == args.save_at_step:
                if args.checkpoint_dir is None:
                    raise ValueError("--save_at_step requires --checkpoint_dir")
                accelerator.wait_for_everyone()
                accelerator.save_state(str(args.checkpoint_dir))
                accelerator.wait_for_everyone()
                saved_and_exiting = True
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        payload = {
            "loss_history": loss_history,
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "num_steps": step,
            "distributed_type": str(accelerator.distributed_type),
            "num_processes": accelerator.num_processes,
            "mixed_precision": str(accelerator.mixed_precision),
            "saved_and_exiting": saved_and_exiting,
            "resumed_from": str(args.resume_from) if args.resume_from is not None else None,
            "world_size_env": int(os.environ.get("WORLD_SIZE", "1")),
        }
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
