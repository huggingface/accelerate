# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Ulysses sequence parallelism on plain PyTorch (`sp_backend="torch"`).

Each rank holds a shard of the sequence. Around every attention call, an all-to-all scatters the heads and gathers the
sequence, the model's own attention runs on the full sequence with `num_heads / sp_size` heads, and a second
all-to-all reverses it. The hook is transformers' `ALL_ATTENTION_FUNCTIONS`: [`UlyssesAttention`] is registered under
the model's own implementation key and only acts on modules registered through [`register_ulysses_attention`].
"""

import weakref
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.distributed as dist


def _all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group) -> torch.Tensor:
    """Split `tensor` along `scatter_dim` across the ranks of `group`, concatenate what comes back along `gather_dim`."""
    world_size = dist.get_world_size(group)
    if tensor.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"Cannot split a tensor of shape {tuple(tensor.shape)} along dim {scatter_dim} across {world_size} "
            "sequence parallel ranks: the size of that dim must be divisible by `sp_size`."
        )
    # Bring the chunk index in front: `all_to_all_single` sends chunk `i` of dim 0 to rank `i` and receives into slot
    # `j` of dim 0 what rank `j` sent, so afterwards dim 0 indexes the pieces of the gathered dim, in rank order.
    shape = list(tensor.shape)
    shape[scatter_dim : scatter_dim + 1] = [world_size, shape[scatter_dim] // world_size]
    tensor = tensor.reshape(shape).movedim(scatter_dim, 0).contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    output = output.movedim(0, gather_dim)
    shape = list(output.shape)
    shape[gather_dim : gather_dim + 2] = [shape[gather_dim] * shape[gather_dim + 1]]
    return output.reshape(shape)


class _SeqAllToAll(torch.autograd.Function):
    """Differentiable all-to-all: the backward pass is the same collective with scatter and gather dims swapped."""

    @staticmethod
    def forward(ctx, group, tensor, scatter_dim, gather_dim):
        ctx.group, ctx.scatter_dim, ctx.gather_dim = group, scatter_dim, gather_dim
        return _all_to_all(tensor, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return None, _all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None


def _gather_along_dim(tensor: torch.Tensor, dim: int, group) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _packed_causal_mask(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Boolean `[batch, 1, seq, seq]` mask for packed sequences: causal within a document, no attention across documents.
    Documents are delimited by `position_ids` restarting, as in `transformers`. Returns `None` when nothing is packed,
    so the caller can rely on `is_causal` instead of materializing a mask.
    """
    first = position_ids[:, :1] - 1
    document_ids = (torch.diff(position_ids, prepend=first, dim=-1) != 1).cumsum(-1)
    if (document_ids[:, -1] == 0).all():
        return None
    seq_len = position_ids.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=position_ids.device).tril()
    same_document = document_ids[:, None, :, None] == document_ids[:, None, None, :]
    return causal & same_document


class UlyssesAttention:
    """
    Attention function for `transformers`' `AttentionInterface` that runs Ulysses sequence parallelism around
    `attention_function`, the implementation registered under `attn_implementation` before us.

    Only the modules registered with [`~UlyssesAttention.register_module`] get the all-to-all treatment; a call from
    any other module goes straight to `attention_function`.
    """

    def __init__(self, attn_implementation: str, attention_function: Callable):
        self.attn_implementation = attn_implementation
        self.attention_function = attention_function
        self._groups = weakref.WeakKeyDictionary()
        self._owners = weakref.WeakKeyDictionary()
        self._model_position_ids = weakref.WeakKeyDictionary()

    def register_module(self, module: torch.nn.Module, group, owner: torch.nn.Module):
        self._groups[module] = group
        self._owners[module] = owner

    def set_position_ids(self, owner: torch.nn.Module, position_ids: torch.Tensor | None):
        """Record the local `position_ids` of the forward `owner` is running, for attention modules that don't get them."""
        self._model_position_ids[owner] = position_ids

    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        group = self._groups.get(module)
        if group is None:
            return self.attention_function(module, query, key, value, attention_mask, **kwargs)

        # Some models only hand their attention layers the rotary embeddings (Qwen2.5-VL, for one); fall back to
        # the `position_ids` seen at the model's forward. Multimodal rotary positions are `[3, batch, seq]`, the
        # text axis is what matters here.
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = self._model_position_ids.get(self._owners[module])
        if position_ids is None:
            raise ValueError(
                "Ulysses sequence parallelism needs `position_ids` in every attention call: after gathering the "
                "sequence, they tell the attention kernel where each token sits and where packed documents start. "
                f"{module.__class__.__name__} was called without them and none were seen at the model's forward. "
                "Pass `position_ids` explicitly, or disable sequence parallelism for this model."
            )
        if position_ids.ndim == 3:
            position_ids = position_ids[0]

        # [batch, heads, local_seq, head_dim] -> [batch, heads / sp_size, seq, head_dim]
        query = _SeqAllToAll.apply(group, query, 1, 2)
        key = _SeqAllToAll.apply(group, key, 1, 2)
        value = _SeqAllToAll.apply(group, value, 1, 2)
        kwargs["position_ids"] = _gather_along_dim(position_ids, 1, group)
        # Flash attention takes precomputed sequence boundaries over `position_ids`; a collator computed them for
        # the local shard, so drop them and let the kernel rebuild them from the gathered positions.
        for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(name, None)
        # Attention sinks (gpt-oss) are one learnable logit per head: keep the ones of the heads this rank now holds.
        s_aux = kwargs.get("s_aux")
        if isinstance(s_aux, torch.Tensor) and s_aux.ndim == 1:
            kwargs["s_aux"] = s_aux.chunk(dist.get_world_size(group))[dist.get_rank(group)]

        # The model built `attention_mask` for its local shard, it does not apply to the gathered sequence. Flash
        # attention derives document boundaries from `position_ids`; SDPA gets a rebuilt mask when sequences are
        # packed and otherwise relies on `is_causal`.
        attention_mask = None
        if self.attn_implementation == "sdpa":
            attention_mask = _packed_causal_mask(kwargs["position_ids"])

        attn_output, attn_weights = self.attention_function(module, query, key, value, attention_mask, **kwargs)

        # [batch, seq, heads / sp_size, head_dim] -> [batch, local_seq, heads, head_dim]
        attn_output = _SeqAllToAll.apply(group, attn_output, 1, 2)
        return attn_output, attn_weights


_ulysses_attention_functions: dict[str, UlyssesAttention] = {}


def register_ulysses_attention(model: torch.nn.Module, group):
    """
    Route the attention calls of `model` (a `transformers` model) through Ulysses sequence parallelism over `group`.

    Replaces the entry of the model's attention implementation in `ALL_ATTENTION_FUNCTIONS` with an
    [`UlyssesAttention`] wrapping the original function, once per implementation and process, and registers the
    modules of `model` with it.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn_implementation = model.config._attn_implementation
    ulysses_attention = _ulysses_attention_functions.get(attn_implementation)
    if ulysses_attention is None:
        if attn_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"Attention implementation `{attn_implementation}` is not registered in transformers' "
                "`ALL_ATTENTION_FUNCTIONS`, so Ulysses sequence parallelism cannot wrap it. "
                f"Valid implementations are {ALL_ATTENTION_FUNCTIONS.valid_keys()}."
            )
        ulysses_attention = UlyssesAttention(attn_implementation, ALL_ATTENTION_FUNCTIONS[attn_implementation])
        ALL_ATTENTION_FUNCTIONS.register(attn_implementation, ulysses_attention)
        _ulysses_attention_functions[attn_implementation] = ulysses_attention
    for module in model.modules():
        ulysses_attention.register_module(module, group, model)
    return ulysses_attention


@contextmanager
def sequence_parallel(
    mesh,
    buffers: list[torch.Tensor] | None = None,
    buffer_seq_dims: list[int] | None = None,
    no_restore_buffers: set[torch.Tensor] | None = None,
):
    """
    Shard `buffers` in place along `buffer_seq_dims` across the ranks of `mesh` for the duration of the context, then
    restore them, except those in `no_restore_buffers`. Same contract as
    `torch.distributed.tensor.experimental.context_parallel`, which [`Accelerator.maybe_context_parallel`] uses for
    `cp_backend="torch"`; this is its counterpart for `sp_backend="torch"`.
    """
    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError("`buffer_seq_dims` must have the same number of elements as `buffers`.")
    for buffer in no_restore_buffers:
        # Cannot use `buffer in buffers`, which would compare tensors elementwise.
        if not any(b is buffer for b in buffers):
            raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

    sp_size, sp_rank = mesh.size(), mesh.get_local_rank()
    original_buffers = [None if buffer in no_restore_buffers else buffer.clone() for buffer in buffers]
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        if buffer.shape[seq_dim] % sp_size != 0:
            raise ValueError(
                f"Cannot shard a buffer of shape {tuple(buffer.shape)} along dim {seq_dim} across {sp_size} sequence "
                "parallel ranks: pad the sequence to a multiple of `sp_size`."
            )
        shard = buffer.chunk(sp_size, dim=seq_dim)[sp_rank].clone()
        buffer.resize_(shard.shape)
        buffer.copy_(shard)

    try:
        yield
    finally:
        for buffer, original in zip(buffers, original_buffers):
            if original is not None:
                buffer.resize_(original.shape)
                buffer.copy_(original)
