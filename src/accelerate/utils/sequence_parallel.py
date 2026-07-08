# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Native (torch-only) Ulysses sequence parallelism; flash kernel or sdpa, no DeepSpeed. Routes the
model through the handler by overwriting its active attn impl (``override_attention``); state on
the handler, no globals.

Ulysses: each rank holds a 1/sp sequence shard (all heads); an all-to-all re-shards q/k/v to
(1/sp heads, full sequence), one local attention runs, the inverse all-to-all restores the layout.
Requires sp_size | num_kv_heads.
"""

import torch
import torch.distributed as dist


def _cu_seqlens_from_position_ids(position_ids):
    """Global packed cu_seqlens + max doc length (doc boundaries are where the position resets to 0).
    Copied from transformers' `prepare_fa_kwargs_from_position_ids` (modeling_flash_attention_utils)."""
    pos = position_ids.reshape(-1)
    cu = torch.cat([(pos == 0).nonzero().view(-1), torch.tensor([pos.numel()], device=pos.device)]).to(torch.int32)
    return cu, int(cu.diff().max())


def _check_kv_divisible(model, sp_size):
    n_kv = model.config.get_text_config().num_key_value_heads
    if n_kv % sp_size:
        raise ValueError(f"sp_size={sp_size} must divide num_key_value_heads={n_kv} for Ulysses.")


# --------------------------------------------------------------------------------------
# Ulysses all-to-all (heads <-> sequence). The permutes emit the flash (b, s, h, d) layout, which
# the attention path then transposes to HF's (b, h, s, d) for the model's own attention. The group
# + sp size are passed through the autograd Functions (stored on ctx for backward), not via globals.
# --------------------------------------------------------------------------------------
def _all_to_all_single(t, group):
    out = torch.empty_like(t)
    dist.all_to_all_single(out, t, group=group)
    return out


def seq_gather(t, group, sp):
    """[b, H, s, d] -> [b, sp*s, H/sp, d]: scatter head groups, gather the sequence."""
    b, n_heads, s, d = t.shape
    t = t.view(b, sp, n_heads // sp, s, d).permute(1, 0, 2, 3, 4).contiguous()
    out = _all_to_all_single(t, group)
    return out.permute(1, 0, 3, 2, 4).reshape(b, sp * s, n_heads // sp, d)


def head_gather(t, group, sp):
    """[b, sp*s, H/sp, d] -> [b, s, H, d]: scatter sequence chunks, gather head groups."""
    b, full, hp, d = t.shape
    s = full // sp
    t = t.view(b, sp, s, hp, d).permute(1, 0, 2, 3, 4).contiguous()
    out = _all_to_all_single(t, group)
    return out.permute(1, 2, 0, 3, 4).reshape(b, s, sp * hp, d)


class GatherSeqScatterHeads(torch.autograd.Function):
    """Ulysses all-to-all: [b, H, s, d] -> [b, S, H/sp, d] (gather the sequence, scatter the
    heads); backward is the inverse exchange."""

    @staticmethod
    def forward(ctx, t, group, sp):
        ctx.group, ctx.sp = group, sp
        return seq_gather(t, group, sp)

    @staticmethod
    def backward(ctx, grad):
        return head_gather(grad, ctx.group, ctx.sp).transpose(1, 2), None, None


class GatherHeadsScatterSeq(torch.autograd.Function):
    """Inverse Ulysses all-to-all: [b, S, H/sp, d] -> [b, s, H, d] (gather the heads, scatter the
    sequence); backward is the inverse exchange."""

    @staticmethod
    def forward(ctx, t, group, sp):
        ctx.group, ctx.sp = group, sp
        return head_gather(t, group, sp)

    @staticmethod
    def backward(ctx, grad):
        return seq_gather(grad.transpose(1, 2), ctx.group, ctx.sp), None, None


class UlyssesAttention:
    """Pure Ulysses: all-to-all to (1/sp heads, full sequence), then the model's OWN attention on
    the gathered sequence, inverse all-to-all. Reuses whatever impl the model loaded (sdpa / pip or
    hub flash) — causal, sliding-window and packed/varlen all handled by HF — so no custom backend
    is needed. In q [b,Hq,s,d] / k,v [b,Hkv,s,d]; out [b,s,Hq,d]. Requires sp_size | Hkv."""

    def __init__(self, sp_group, attn_fn):
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.attn_fn = attn_fn  # the model's original HF attention, captured before we override it
        self.cu_seqlens = None
        self.max_seqlen = None

    def set_varlen(self, cu_seqlens, max_seqlen):
        self.cu_seqlens, self.max_seqlen = cu_seqlens, max_seqlen

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        group, sp = self.sp_group, self.sp_size
        # heads -> full sequence, then to HF's [b, h, S, d] layout
        q = GatherSeqScatterHeads.apply(query, group, sp).transpose(1, 2)
        k = GatherSeqScatterHeads.apply(key, group, sp).transpose(1, 2)
        v = GatherSeqScatterHeads.apply(value, group, sp).transpose(1, 2)
        # Drop the LOCAL-shard kwargs (position_ids + flash varlen indices keyed to the pre-gather
        # length); re-inject the GLOBAL packed cu_seqlens for the full sequence the all-to-all rebuilt.
        for k_ in ("position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(k_, None)
        if self.cu_seqlens is not None:
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = self.cu_seqlens.to(q.device)
            kwargs["max_length_q"] = kwargs["max_length_k"] = self.max_seqlen
        out, _ = self.attn_fn(module, q, k, v, None, **kwargs)
        return GatherHeadsScatterSeq.apply(out, group, sp), None


def enable_ulysses_sp(model, sp_group):
    """Register pure Ulysses SP on ``model`` over ``sp_group``, reusing the model's own attention.
    Returns the handler (a dataloader pushes per-step varlen via ``handler.set_varlen``)."""
    # Ulysses swaps the model's HF attention, so it must be a transformers model (same check as ALST).
    if not hasattr(model, "config"):
        raise ValueError(
            f"Ulysses SP expects a HF Transformers model with a `config` attribute, got {type(model).__name__}."
        )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    _check_kv_divisible(model, dist.get_world_size(sp_group))
    # Capture the model's ORIGINAL attention. If SP was already enabled (re-prepare, second model),
    # the registered entry is our own handler — unwrap it so we don't wrap a wrapper (double all-to-all).
    attn_impl_str = model.config.get_text_config()._attn_implementation
    current = ALL_ATTENTION_FUNCTIONS[attn_impl_str]
    base = current.attn_fn if isinstance(current, UlyssesAttention) else current
    handler = UlyssesAttention(sp_group, base)
    ALL_ATTENTION_FUNCTIONS[attn_impl_str] = handler
    return handler


# --------------------------------------------------------------------------------------
# Sequence-sharding primitive: shared by the dataloader adapter (SFT) and by callers that need to
# shard an already-materialized batch at the forward (e.g. generated GRPO rollouts).
# --------------------------------------------------------------------------------------
def shard_sequence_batch(batch, shard_group, attention=None, seq_dim=1, ignore_index=-100):
    """Contiguously shard a batch's sequence over ``shard_group`` for native Ulysses SP.

    ``prepare_data_loader`` already hands every shard rank the SAME sample (its data-parallel
    sharding divides out tp*cp*sp), so this only does the sequence split: it builds the GLOBAL
    ``position_ids``/``shift_labels`` (computed on the full sequence so shard boundaries keep
    next-token alignment), pushes the GLOBAL packed ``cu_seqlens`` onto ``attention`` via
    ``set_varlen`` (so packed/varlen batches take the flash-varlen path; ``cu_seqlens`` is NOT
    sharded — it indexes the full sequence the all-to-all reconstitutes on each rank), then shards
    ``input_ids``/``labels``/``position_ids``/``shift_labels`` along ``seq_dim``.

    Returns the local shard as a new dict (the input ``batch`` is not mutated). The caller is
    responsible for ensuring the global sequence length is divisible by the shard world size.

    Args:
        batch: mapping with at least ``input_ids`` (and optionally ``labels``/``position_ids``).
        shard_group: the sequence-shard (``sp``) process group.
        attention: the Ulysses attention handler; packed batches push the GLOBAL cu_seqlens onto it.
            ``None`` to disable varlen.
        seq_dim: the sequence dimension. ignore_index: final-token / pad label id.
    """
    rank, world = dist.get_rank(shard_group), dist.get_world_size(shard_group)
    batch = dict(batch)
    # 1. global position_ids
    if "position_ids" not in batch and "input_ids" in batch:
        ids = batch["input_ids"]
        pos = torch.arange(ids.shape[seq_dim], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        batch["position_ids"] = pos.contiguous()
    # 2. global shift_labels (shift before sharding)
    if "labels" in batch and "shift_labels" not in batch:
        labels = batch["labels"]
        shift = torch.full_like(labels, ignore_index)
        shift[..., :-1] = labels[..., 1:]
        batch["shift_labels"] = shift
    # 3. packed/varlen: derive GLOBAL cu_seqlens from the (pre-shard) global position_ids and push
    # onto the attention handler (packed iff position_ids reset more than once).
    if attention is not None:
        packed = "position_ids" in batch and int((batch["position_ids"][0] == 0).sum()) > 1
        attention.set_varlen(*(_cu_seqlens_from_position_ids(batch["position_ids"]) if packed else (None, None)))
    # 4. contiguous sequence shard across the group
    if world > 1:
        for key in ("input_ids", "labels", "position_ids", "shift_labels"):
            t = batch.get(key)
            if t is not None:
                batch[key] = t.chunk(world, dim=seq_dim)[rank].contiguous()
    return batch


# --------------------------------------------------------------------------------------
# Dataloader adapter: replicate-then-shard the sequence across the shard group.
# --------------------------------------------------------------------------------------
class SequenceShardingDataLoader:
    """Sequence-sharding DataLoader wrapper for native Ulysses SP (torch analogue of DeepSpeed's
    ``UlyssesSPDataLoaderAdapter``). accelerate's ``prepare_data_loader`` already hands every sp
    rank the SAME sample (its data-parallel sharding divides out tp*cp*sp), so per batch this just
    builds global ``position_ids``/``shift_labels`` (shifted on the full sequence so shard
    boundaries keep next-token alignment) and shards the sequence contiguously over the sp group.
    Loss normalization is left to the trainer/accelerate.

    Args:
        dataloader: the accelerate-prepared DataLoader to wrap (must hand sp ranks the same sample).
        shard_group: the sequence-shard (``sp``) group.
        attention: the attention handler; packed batches push the GLOBAL cu_seqlens onto it
            (``set_varlen``). None to disable varlen.
        seq_dim / ignore_index: sequence dim / final-token label id.
    """

    def __init__(self, dataloader, shard_group, attention=None, seq_dim=1, ignore_index=-100):
        self.dataloader = dataloader
        self.attention = attention
        self.seq_dim = seq_dim
        self.ignore_index = ignore_index
        self.shard_group = shard_group
        self.rank = dist.get_rank(shard_group)
        self.world = dist.get_world_size(shard_group)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield self._process(dict(batch))

    def _process(self, batch):
        # accelerate's prepare_data_loader already hands every sp rank the SAME sample, so this just
        # builds the global position_ids / shift_labels and shards the sequence over the sp group.
        return shard_sequence_batch(
            batch, self.shard_group, attention=self.attention, seq_dim=self.seq_dim, ignore_index=self.ignore_index
        )
