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
Verify native ("accelerate") Ulysses sequence parallelism end-to-end. It's engine-agnostic, so the
same script runs under DDP, FSDP2, and DeepSpeed ZeRO (`--engine`), optionally composed with data
parallelism (the remaining `world // sp` becomes dp) and with packed/varlen sequences (`--packed`).

It checks that `prepare()`:
  * registers the Ulysses attention on the model, and
  * swaps the dataloader for a `SequenceShardingDataLoader` that gives every sp rank the SAME sample,
    shards its sequence over the sp group, builds `position_ids`/`shift_labels`, and (packed) pushes
    the global cu_seqlens onto the attention handler,
then runs a few training steps with finite loss.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ParallelismConfig, set_seed
from accelerate.utils.sequence_parallel import SequenceShardingDataLoader


MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # kv_heads=4, divisible by sp_size
SEQLEN = 64  # must be divisible by sp_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ddp", "fsdp", "deepspeed"], default="ddp")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--sp-size", type=int, default=2)
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
    # ddp/fsdp from the args above; deepspeed engine comes from `accelerate launch --use_deepspeed`.
    accelerator = Accelerator(parallelism_config=ParallelismConfig(**pc_kwargs), fsdp_plugin=fsdp_plugin)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
    model.config.use_cache = False
    vocab = model.config.vocab_size

    # Each sample is a CONSTANT token (i+1) so that a sequence-sharded batch still reveals which
    # sample it came from -> we can assert every sp rank received the SAME sample. Packed: two docs
    # per sample, with position_ids resetting at the doc boundary.
    n = 4 * dp
    samples = torch.stack([torch.full((SEQLEN,), i % (vocab - 1) + 1, dtype=torch.long) for i in range(n)])
    if args.packed:
        half = SEQLEN // 2
        positions = torch.cat([torch.arange(half), torch.arange(SEQLEN - half)]).unsqueeze(0).expand(n, -1)
        ds = torch.utils.data.TensorDataset(samples, positions.contiguous())

        def collate(b):
            ids, pos = torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])
            return {"input_ids": ids, "labels": ids.clone(), "position_ids": pos}
    else:
        ds = torch.utils.data.TensorDataset(samples)

        def collate(b):
            ids = torch.stack([x[0] for x in b])
            return {"input_ids": ids, "labels": ids.clone()}

    dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    sp_group = accelerator.torch_device_mesh["sp"].get_group()
    sp_world = torch.distributed.get_world_size(sp_group)
    unwrapped = accelerator.unwrap_model(model)

    # --- the dataloader did its job ---
    assert isinstance(dl, SequenceShardingDataLoader), f"dataloader not wrapped: {type(dl)}"
    batch = next(iter(dl))
    assert batch["input_ids"].shape[1] == SEQLEN // sp_world, "sequence was not sharded over sp"
    assert "shift_labels" in batch and "position_ids" in batch, "missing shift_labels / position_ids"
    # every sp rank must hold the SAME sample (constant token -> compare the value across the sp group)
    tok = batch["input_ids"].flatten()[:1].clone()
    gathered = [torch.empty_like(tok) for _ in range(sp_world)]
    torch.distributed.all_gather(gathered, tok, group=sp_group)
    assert all(torch.equal(t, gathered[0]) for t in gathered), f"sp ranks got DIFFERENT samples: {gathered}"
    if args.packed:
        assert accelerator._sp_attention.cu_seqlens is not None, "packed batch did not set cu_seqlens"

    # --- gradients flow + reduce correctly: overfit this one (sequence-sharded) batch and require
    # the loss to drop. Step-0 alone would only test the forward; a multi-step decrease is what
    # catches broken gradient flow / sp-dp reduction (e.g. if sp ranks weren't kept in sync). ---
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    losses = []
    for step in range(8):
        out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"])
        loss = unwrapped.loss_function(
            logits=out.logits, labels=None, shift_labels=batch["shift_labels"], vocab_size=vocab
        )
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        accelerator.print(f"[{args.engine} dp{dp}xsp{sp}{' packed' if args.packed else ''}] step {step}: {loss.item():.4f}")

    assert all(loss == loss for loss in losses), f"non-finite loss: {losses}"  # NaN check
    assert losses[-1] < losses[0], f"loss did not decrease over steps: {losses}"

    accelerator.end_training()


if __name__ == "__main__":
    main()
