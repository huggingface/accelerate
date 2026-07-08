# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Verify native ("accelerate") Ulysses sequence parallelism end-to-end. Engine-agnostic: the same
script runs under DDP, FSDP2, and DeepSpeed ZeRO (`--engine`), optionally with data parallelism
(the remaining `world // sp` becomes dp) and packed/varlen sequences (`--packed`).

Correctness is checked by **parity**: Ulysses only splits the sequence, so for the same dp the loss
must not depend on sp. We compare the *global mean loss* (token-weighted over the whole world, which
is invariant to how the global batch is split across dp and sp) between two runs:
  * `--save-ref FILE` : an `sp=1` reference run (full sequences) saves its loss trajectory.
  * `--ref FILE`      : the `sp>1` run loads it and asserts a per-step match.
e.g. `dp=2,sp=1` on 2 GPUs vs `dp=2,sp=2` on 4 GPUs. Every sp run also checks the auto-wrapped
`SequenceShardingDataLoader` (sequence sharded, shift_labels / position_ids built, packed cu_seqlens).
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ParallelismConfig, set_seed
from accelerate.utils.sequence_parallel import SequenceShardingDataLoader


MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # kv_heads=4, divisible by sp_size
SEQLEN = 64  # must be divisible by sp_size
STEPS = 8
LR = 1e-3
PARITY_TOL = 1e-2  # sp>1 vs sp=1 global-loss trajectory; tolerates fp32 reduction-order drift


def shifted(ids, ignore_index=-100):
    s = torch.full_like(ids, ignore_index)
    s[..., :-1] = ids[..., 1:]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--save-ref", default=None, help="sp=1 reference: save the loss trajectory here")
    parser.add_argument("--ref", default=None, help="compare the loss trajectory against this reference file")
    args = parser.parse_args()

    set_seed(42)
    world = int(os.environ["WORLD_SIZE"])
    sp, dp = args.sp_size, world // args.sp_size

    pc_kwargs = {"sp_size": sp, "sp_backend": "accelerate"}
    pc_kwargs["dp_replicate_size" if args.engine == "ddp" else "dp_shard_size"] = dp
    fsdp_plugin = None
    if args.engine == "fsdp":
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
        )
    accelerator = Accelerator(parallelism_config=ParallelismConfig(**pc_kwargs), fsdp_plugin=fsdp_plugin)
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
    model.config.use_cache = False
    vocab = model.config.vocab_size

    # One batch per sample; deterministic (set_seed) so the same dp gets the same samples regardless
    # of sp. Packed: two docs per sample, position_ids resetting at the boundary.
    n = 4 * dp
    samples = torch.randint(1, vocab, (n, SEQLEN))
    if args.packed:
        half = SEQLEN // 2
        positions = torch.cat([torch.arange(half), torch.arange(SEQLEN - half)])[None].expand(n, -1).contiguous()
        ds = torch.utils.data.TensorDataset(samples, positions)
        collate = lambda b: {  # noqa: E731
            "input_ids": torch.stack([x[0] for x in b]),
            "labels": torch.stack([x[0] for x in b]),
            "position_ids": torch.stack([x[1] for x in b]),
        }
    else:
        ds = torch.utils.data.TensorDataset(samples)
        collate = lambda b: {"input_ids": torch.stack([x[0] for x in b]), "labels": torch.stack([x[0] for x in b])}  # noqa: E731

    dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    unwrapped = accelerator.unwrap_model(model)

    if sp > 1:  # native SP enabled -> the dataloader is auto-wrapped and the sequence is sharded
        sp_world = dist.get_world_size(accelerator.torch_device_mesh["sp"].get_group())
        assert isinstance(dl, SequenceShardingDataLoader), f"dataloader not wrapped: {type(dl)}"
        probe = next(iter(dl))
        assert probe["input_ids"].shape[1] == SEQLEN // sp_world, "sequence was not sharded over sp"
        assert "shift_labels" in probe and "position_ids" in probe, "missing shift_labels / position_ids"
        if args.packed:
            assert accelerator._sp_attention.cu_seqlens is not None, "packed batch did not set cu_seqlens"

    # First batch (sequence-sharded when sp>1; full when sp=1 -> build shift_labels / position_ids).
    batch = {k: v.to(device) for k, v in next(iter(dl)).items()}
    if "position_ids" not in batch:
        batch["position_ids"] = torch.arange(batch["input_ids"].shape[1], device=device)[None].expand_as(batch["input_ids"])
    if "shift_labels" not in batch:
        batch["shift_labels"] = shifted(batch["labels"])

    def global_mean_loss(shard_loss, shift_labels):
        # token-weighted mean over the WHOLE world == the global mean CE, invariant to how the global
        # batch is split across dp (different samples) and sp (sequence shards).
        if world == 1 or not dist.is_initialized():
            return shard_loss.item()
        good = (shift_labels != -100).sum().float()
        losses = [torch.empty_like(shard_loss) for _ in range(world)]
        goods = [torch.empty_like(good) for _ in range(world)]
        dist.all_gather(losses, shard_loss.detach())
        dist.all_gather(goods, good)
        return (sum(loss * g for loss, g in zip(losses, goods)) / sum(goods)).item()

    losses = []
    for step in range(STEPS):
        out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"])
        loss = unwrapped.loss_function(
            logits=out.logits, labels=None, shift_labels=batch["shift_labels"], vocab_size=vocab
        )
        losses.append(global_mean_loss(loss, batch["shift_labels"]))
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.print(f"[{args.engine} dp{dp}xsp{sp}{' packed' if args.packed else ''}] step {step}: {losses[-1]:.5f}")

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    if args.save_ref and accelerator.is_main_process:
        json.dump(losses, open(args.save_ref, "w"))
    if args.ref:
        ref = json.load(open(args.ref))
        worst = max(abs(a - b) for a, b in zip(ref, losses))
        assert worst < PARITY_TOL, f"sp={sp} loss != sp=1 reference (max abs diff {worst:.4g}): {losses} vs {ref}"

    accelerator.end_training()


if __name__ == "__main__":
    main()
