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
Test script for verifying Ulysses sequence parallelism on FSDP2 (`sp_backend="torch"`).

Runs a forward/backward with the sequence sharded across the `sp` ranks and checks logits and gradients against a
plain, unsharded forward/backward of the same model on the same rank.
"""

import argparse

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ParallelismConfig, set_seed
from accelerate.utils.ulysses import _all_to_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp_size", type=int, default=2)
    parser.add_argument("--dp_shard_size", type=int, default=1)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--model_name_or_path", type=str, default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--packed", action="store_true", help="Pack two documents per sample (position_ids restart)")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--sliding_window", type=int, default=None, help="Override the model's sliding window")
    parser.add_argument(
        "--stale_flash_attn_kwargs",
        action="store_true",
        help="Pass flash attention sequence boundaries computed for the local shard, as a padding-free collator would",
    )
    return parser.parse_args()


def test_all_to_all_roundtrip(sp_group):
    """Scatter heads / gather sequence, then the reverse, must give the input back."""
    tensor = torch.randn(2, 8, 6, 4, device="cuda")
    gathered = _all_to_all(tensor, scatter_dim=1, gather_dim=2, group=sp_group)
    assert gathered.shape == (2, 8 // dist.get_world_size(sp_group), 6 * dist.get_world_size(sp_group), 4)
    restored = _all_to_all(gathered, scatter_dim=2, gather_dim=1, group=sp_group)
    torch.testing.assert_close(restored, tensor)


def main():
    args = parse_args()
    set_seed(42)
    dtype = getattr(torch, args.dtype)

    parallelism_config = ParallelismConfig(dp_shard_size=args.dp_shard_size, sp_size=args.sp_size, sp_backend="torch")
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, auto_wrap_policy="transformer_based_wrap", state_dict_type="SHARDED_STATE_DICT"
    )
    accelerator = Accelerator(parallelism_config=parallelism_config, fsdp_plugin=fsdp_plugin)
    mesh = accelerator.torch_device_mesh
    assert "sp" in mesh.mesh_dim_names, mesh.mesh_dim_names
    assert mesh["dp_shard_cp"].size() == args.dp_shard_size * args.sp_size, "sp must be folded into the FSDP mesh"
    sp_group = mesh["sp"].get_group()
    sp_rank = mesh["sp"].get_local_rank()

    test_all_to_all_roundtrip(sp_group)

    # Reference model: same weights, plain attention, whole sequence on every rank. Forwards run with
    # `use_cache=False` as in training: with a cache, transformers' sdpa ignores the packing carried by `position_ids`.
    model_kwargs = dict(attn_implementation=args.attn_implementation, dtype=dtype)
    if args.sliding_window is not None:
        model_kwargs["sliding_window"] = args.sliding_window
    # Vision-language models train on text only here; they take the text-only path with `input_ids` alone
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_cls = (
        AutoModelForImageTextToText if getattr(config, "vision_config", None) is not None else AutoModelForCausalLM
    )
    reference = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs).to(accelerator.device)
    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    # tiny test checkpoints may lack a weight (Qwen2.5-VL's lm_head), which is then initialized at random per load
    model.load_state_dict(reference.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Each data parallel rank gets its own batch; sequence parallel ranks share it.
    generator = torch.Generator().manual_seed(accelerator.data_parallel_shard_rank)
    input_ids = torch.randint(
        10, reference.config.get_text_config().vocab_size, (1, args.seq_len), generator=generator
    )
    input_ids = input_ids.to(accelerator.device)
    if args.packed:
        half = args.seq_len // 2
        position_ids = torch.cat([torch.arange(half), torch.arange(args.seq_len - half)]).unsqueeze(0)
    else:
        position_ids = torch.arange(args.seq_len).unsqueeze(0)
    position_ids = position_ids.to(accelerator.device)
    labels = input_ids.clone()
    shift_labels = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)

    # Reference forward/backward over the full sequence, averaged over the data parallel ranks by hand.
    reference_out = reference(input_ids=input_ids, position_ids=position_ids, labels=labels, use_cache=False)
    reference_out.loss.backward()
    reference_grad = reference.get_input_embeddings().weight.grad.clone()
    if args.dp_shard_size > 1:
        dist.all_reduce(reference_grad, op=dist.ReduceOp.AVG, group=mesh["dp_shard"].get_group())

    # `labels` makes the model compute a loss, `shift_labels` are the pre-shifted ones it then actually uses: shifting
    # inside a shard would drop the token that sits on the next rank.
    buffers = [input_ids, position_ids, labels, shift_labels]
    with accelerator.maybe_context_parallel(
        buffers=buffers, buffer_seq_dims=[1, 1, 1, 1], no_restore_buffers=set(buffers)
    ):
        assert input_ids.shape[1] == args.seq_len // args.sp_size, input_ids.shape
        # A padding-free collator computes the flash attention boundaries for the shard it sees, which is the local
        # one. Attention must rebuild them for the gathered sequence rather than use these.
        flash_attn_kwargs = {}
        if args.stale_flash_attn_kwargs:
            from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )
            flash_attn_kwargs = {
                "cu_seq_lens_q": cu_seq_lens_q,
                "cu_seq_lens_k": cu_seq_lens_k,
                "max_length_q": max_length_q,
                "max_length_k": max_length_k,
            }
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            shift_labels=shift_labels,
            use_cache=False,
            **flash_attn_kwargs,
        )
        # Every rank gets the mean loss of its own tokens. Weight the per-rank losses by their token counts through a
        # differentiable all_gather: FSDP then averages the gradients across `dp_shard_cp`, which gives the gradient
        # of the global mean loss, as if the whole sequence had been processed on one rank.
        losses_per_rank = torch.distributed.nn.functional.all_gather(outputs.loss, group=sp_group)
        good_tokens = (shift_labels != -100).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        loss = sum(loss * tokens for loss, tokens in zip(losses_per_rank, good_tokens_per_rank))
        loss = loss / sum(good_tokens_per_rank)
        accelerator.backward(loss)

    tolerance = {"atol": 1e-4, "rtol": 1e-4} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 2e-2}
    # Logits of the local shard must match the corresponding slice of the reference.
    local_reference_logits = reference_out.logits.chunk(args.sp_size, dim=1)[sp_rank]
    torch.testing.assert_close(outputs.logits.float(), local_reference_logits.float(), **tolerance)
    torch.testing.assert_close(loss.float(), reference_out.loss.float(), **tolerance)
    grad = model.get_input_embeddings().weight.grad.full_tensor()
    torch.testing.assert_close(grad.float(), reference_grad.float(), **tolerance)
    accelerator.print(f"Ulysses SP ({args.attn_implementation}, packed={args.packed}) matches the reference: {loss=}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
