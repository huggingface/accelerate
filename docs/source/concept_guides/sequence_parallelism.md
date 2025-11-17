<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Sequence parallel in 🤗`accelerate`

This guide will cover basics of using sequence parallelism in 🤗`accelerate`.

See also the very related [Context Parallellism](./context_parallelism.md).

## Why sequence parallelism?

With the advent of large language models, and recently reasoning models, the sequence length has been growing rapidly. This, combined with quadratic memory complexity of attention, has led to a need for more efficient ways to train models with long sequences.
With sequence length of 128k, the memory requirement of the attention matrix is `128k * 128k * 2 bytes * num_heads = ~32 GB * num_heads` for `bf16` precision, given vanilla attention implementation. Granted, with usage of `flash attention` or `SDPA` which do not materialize these attention weights, this decreases drastically, but the growth in memory requirements is still considerable.

Ulysses Sequence parallelism allows us to shard the inputs to the attention computation along the sequence dimension and compute the attention normally, but using only a slice of attention heads on each GPU. With this, we can train models with long sequences, with a few more tools, scaling to 15M+ sequence length. To see how to augment Ulysses SP with TiledMLP, Liger-Kernel, Activation checkpoint offload to cpu and a few other tricks pleae refer to the paper: [Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences](https://arxiv.org/abs/2506.13996).

## How is Ulysses SP different from FSDP CP

In the document [Context Parallellism](./context_parallelism.md) you can learn about deploying another technology called Context Parallelism, which too slices on the sequence dimension but uses Ring Attention instead of slicing on the head dimension.

The following articles go into a very detailed explanation of the differences between the two technologies:
- https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/
- https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention

A quick summary adapting from one of the articles:
- Ulysses SP has a relatively low communication overhead, but is limited by the number of Attention Heads and thus it has certain requirements for network topology (number of attention heads has has to be divisible by the number of participating gpus for a single replica). All-to-all communication is sensitive to latency and it requires Deepspeed.
- FSDP CP Ring-Attention's P2P ring communication has no aforementioned divisibilty requirements, but has a higher communication volume.

Finally it should be possible to combine SP + CP as explained in the paper [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719) to support an even longer sequence length, albeit this is not yet integrated into 🤗`accelerate`.


## Supported sequence parallelism backends

Currently the only sequence parallelism backend is `deepspeed`, which comes from the modernized Ulysses SP which is part of the [Arctic Long Sequence Training technology](https://arxiv.org/abs/2506.13996). There is also a [tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/) should you want to integrate it into your own code directly.

## How to use sequence parallelism?

```diff
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

parallelism_config = ParallelismConfig(
+     sp_backend="deepspeed",
+     sp_size=4,
+     sp_handler=DeepSpeedSequenceParallelConfig(
+         sp_seq_length_is_variable: true,
+         sp_attn_implementation="sdpa",
+     ),
+ )

accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```

As with any other feature in 🤗`accelerate`, you can enable sequence parallelism also by passing the corresponding flags to `accelerate launch`. In this case, it's no different:

```bash
accelerate launch --parallelism-config-sp-size 8  ...
```

> [!Tip]
> You can also set the `sp_size` and other configuration in the `accelerate config` command, which will save them in your `accelerate` configuration file, so you don't have to pass them every time you launch your script.

> [!Tip]
> sequence parallelism combines with data parallelism. It doesn't require additional GPUs.
> So if you have 8 gpus you can do: `--parallelism-config-dp-shard-size 8 --parallelism-config-sp-size 8`. Or you can use the `ParallelismConfig` class to set them programmatically.


## ALST/Ulysses SP backend configuration

ALST/UlyssesSP implements sequence parallelism using attention head parallelism, as explained in [this paper](https://arxiv.org/abs/2506.13996). For simplicity, we reuse the concept and setup of sequence parallelism, which, from the user's perspective, is the same: multiple GPUs are used to process a single batch.

To give a sense of what ALST made possible - it allowed us to train in bf16 with 500K tokens on a single H100 GPU, 3.7M on a single node, and 15M on Llama-8B using just four nodes. This feature of HF Accelerate enables only 1 of the 3 ALST components, so the achievable sequence length will be smaller. You'd want TiledMLP, Activation checkpoint offload to CPU, and a few other things enabled to get the full power of ALST. For details, please refer to [this tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/).

To configure the `deepspeed` backend:

```python
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length=256,
        sp_seq_length_is_variable=True,
        sp_attn_implementation="sdpa",
    ),
)
accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```

- `sp_backend`: set to `deepspeed` here
- `sp_size` is the degree of the sequence parallelism - in the above example it's 4, therefore 4 gpus will be used to process a single batch (while doing DP=4 over the same gpus)
- `sp_seq_length` and `sp_seq_length_is_variable` are used to deal with sequence lengths. If `sp_seq_length_is_variable=True` the backend will work with a sequence length that may change between batches, in which case `sp_seq_length` value can be set to anything divisible by the sequence parallel degree or not set at all. In this case on every `forward` the sequence variables will be derived from input. If `False` then `seq_length` needs to match the batch's sequence length dimension, which then will have to be padded to be always the same. The default is `True`.
- `sp_attn_implementation` is one of `sdpa`, `flash_attention_2` or `flash_attention_3`. This sequence parallel implementation uses `position_ids` instead of `attention_mask` therefore, `eager` can't work here until it supports working with `position_ids`. Also, please note that `sdpa` doesn't handle multiple samples combined into one correctly; it will attend to the whole sample as one. If the samples aren't combined, `sdpa` will work correctly. Therefore, Flash Attention should be the ideal choice as it always works.

Instead of setting these values in `DeepSpeedSequenceParallelConfig` object, you can also use the environment variables to accomplish the same - here they are correspondingly to the end of the list above.
- `PARALLELISM_CONFIG_SP_BACKEND`
- `PARALLELISM_CONFIG_SP_SEQ_LENGTH`
- `PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE`
- `PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION`

If not passed in the code, `sp_size` can be set via `--parallelism_config_sp_size` CLI argument. Same for other arguments. You can also do the accelerate config file style config, e.g., for 2 GPUs:

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 2
parallelism_config:
  parallelism_config_sp_size: 2
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: sdpa

```

As mentioned earlier Ulysses sequence parallelism is normally overlayed with data parallelism - same ranks are used for feeding unique data streams and also perform Ulysses Sequence Parallelism. But you could also create replicas like so:

```python
parallelism_config = ParallelismConfig(
    dp_shard_size=2,
    sp_backend="deepspeed",
    sp_size=2,
    sp_handler=DeepSpeedSequenceParallelConfig(...),
)
```
Here we use 4 gpus, with 2 sequence parallelism replicas. Deepspeed-ZeRO is what drives the data parallelism here.

Please note that a lot of magic is hidden inside [UlyssesSPDataLoaderAdapter](https://github.com/deepspeedai/DeepSpeed/blob/64c0052fa08438b4ecf4cae30af15091a92d2108/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L442). It's used behind the scenes, wrapping your original DataLoader object, but you should be aware of it should you run into any problems. It also automatically injects the correct `shift_labels` into the batch dictionary, before the batch gets sharded across the participating ranks.

Now the only remaining piece to start using ALST/UlyssesSP is to aggregate the loss across ranks using a differentiable `all_gather` to get the grads right. The following code does it, while also excluding any masked out with `-100` tokens, to get the correct average:

```python
sp_size = parallelism_config.sp_size if parallelism_config is not None else 1
if sp_size > 1:
    sp_group = accelerator.torch_device_mesh["sp"].get_group()
    sp_world_size = parallelism_config.sp_size

# Normal training loop
for iter, batch in enumerate(dl):
    optimizer.zero_grad()

    batch = move_to_device(batch, model.device)
    outputs = model(**batch)

    # only if not using liger-kernel
    shift_labels = batch["shift_labels"]
    loss = unwrapped_model.loss_function(
        logits=outputs.logits,
        labels=None,
        shift_labels=shift_labels,
        vocab_size=unwrapped_model.config.vocab_size,
    )

    if sp_size > 1:
        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
        # special dealing with SFT that has prompt tokens that aren't used in loss computation
        good_tokens = (shift_labels != -100).view(-1).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(
            good_tokens, group=sp_group
        )
        total_loss = sum(
            losses_per_rank[rank] * good_tokens_per_rank[rank]
            for rank in range(sp_world_size)
        )
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / max(total_good_tokens, 1)

    if rank == 0: accelerator.print(f"{iter}: {loss=}")
    accelerator.log(dict(train_loss=loss, step=iter))

    accelerator.backward(loss)
    optimizer.step()
```

If you use [Liger Kernel](https://github.com/linkedin/Liger-Kernel) it already knows how to handle `shift_labels` so you don't need to go through manual loss calculation, just calling `model(**batch)` will already get the `loss` calculated and done in a very memory-efficient way. If you didn't know about Liger-Kernel - it's highly recommended to be used especially for long sequence length, since it liberates a lot of working GPU memory that can be used for handling longer sequences. For example, it performs a fused logit-loss computation, never manifesting the full logits tensor in memory.

If you want to see what HF Accelerate did behind the scenes please read [this full integration tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/).

For an example of an Accelerate training loop with enabled ALST/UlyssesSP see [examples/alst_ulysses_sequence_parallelism](https://github.com/huggingface/accelerate/blob/main/examples/alst_ulysses_sequence_parallelism).

[!Warning]
> This API is quite new and still in its experimental stage. While we strive to provide a stable API, some small parts of the public API may change in the future.

Since this is a Deepspeed backend the usual Deepspeed configuration applies, so you can combine sequence parallelism with optimizer states and/or weights offloading as well to liberate more gpu memory and enable an even longer sequence length. This technology has been tested to work with DeepSpeed ZeRO stage 2 and 3.

