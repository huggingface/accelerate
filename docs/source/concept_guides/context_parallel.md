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

# Context Parallel in 🤗`accelerate`

This guide will cover basics of using context parallelism in 🤗`accelerate`, for the more curious readers, we will also cover some technicalities in the later sections.

## Why context parallelism?

With the advent of large language models, and recently reasoning models, the sequence length has been growing rapidly. This, combined with quadratic memory complexity of attention, has lead to a need for more efficient ways to train models with long sequences.
With sequence length of 128k, the memory requirement of the attention matrix is `128k * 128k * 2 * num_heads = ~32 GB * num_heads` for `bf16` precision, given vanilla attention implementation. Granted, with usage of `flash attention` or `SDPA` which do not materialize these attention weights, this decreases drastically, but the growth in memory requirements is still considerable.

Context parallelism allows us to shard the inputs to the attention computation along the sequence dimension and compute the attention in parallel on multiple GPUs. With this, we can train models with long sequences, scaling potentially to 1M+ sequence length.


## How to use context parallelism?

As with any other feature in 🤗`accelerate`, enabling context parallelism is as simple as passing the corresponding flags to `accelerate launch`.
In this case, it's no different:

```bash
accelerate launch --context-parallel-size 8 --context-parallel-shard-rotation [allgather|alltoall] ...
```

Context parallelism is tightly coupled (for now) with `FSDP2`, which you can learn more about in the [FSDP2 introduction](fsdp1_vs_fsdp2.md). Meaning, context parallelism is applied only if `FSDP2` is enabled.
You can also enable context parallelism programatically, by passing it in the `FullyShardedDataParallelPlugin` constructor:

```diff
from accelerate.utils import FullyShardedDataParallelPlugin

plugin = FullyShardedDataParallelPlugin(
    ...
    fsdp_version=2,
+   context_parallel_size=8,
+   context_parallel_shard_rotation="allgather",
)
accelerator = Accelerator(fsdp_plugin=plugin)
```

After enabling context parallelism with the methods mentioned above, you can then apply it to your training loop. We provide a thin wrapper around [`torch.distributed.tensor.experimental.context_parallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel) that you can use in your training loop, that abstracts some of the complexity of using it (more on this later).
You can use it as follows:

```python
for batch in dataloader:
    with accelerator.context_parallel(
        buffers=[batch["input_ids"], batch["attention_mask"]],
        buffer_seq_dims=[1, 1],
        no_restore_buffers={batch["input_ids"]},
    ):
        outputs = model(batch)
        ...
```

> [!Warning]
> This context manager has to be recreated with each training step, as shown in the example above. It's crucial to do so.

## Accelerate's interface

The context manager takes a few arguments, that are used to configure the context parallelism.

- `buffers`: This is a list of tensors that are to be sharded across the sequence dimension. These tensors are usually input ids, labels and attention mask.
- `buffer_seq_dims`: This is a list of integers, that specify the sequence dimension of the buffers, in the order of the `buffers` list.
- `no_restore_buffers`: The implementation of context parallelism modifies the buffers in-place, converting them to `torch.distributed.tensor.Dtensor`s. After the context manager is exited, a communication kernel would need to be launched to restore the buffers to their original state (usually all-gather). This takes some time, so it is reccomended to pass the same arguments as to the `buffers` argument, to avoid unnecessary communication, unless you are sure that you need to use the buffers after the context manager is exited.

## Configurable options
Accelerate provides only a few options to configure context parallelism, which are:

- `context_parallel_size`: The number of ranks to shard the inputs to the attention computation across the sequence dimension.
- `context_parallel_shard_rotation`: The rotation method to use for the shards. We strongly reccomend keeping this as `"allgather"`, as it's very likely it will outperform `"alltoall"` in most cases.

Context parallel size is rather self-explanatory, it's the number of ranks across which the inputs are to be-sharded.
Context parallel shard rotation defines how the shards of the inputs are rotated across ranks. We'll cover the 2 options in more detail in the next section.

You can see an end-to-end example in the [FSDP2 context parallel example](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/fsdp2_context_parallel.py) file, where you can train an 8B model with 128k sequence length on 8x H100 SXM GPUs. Using multi-node training, you can scale this to 1M+ sequence length on 64x H100 SXM GPUs.

## Technical details

> [!Tip]
> This section is fairly technical, so if you don't need to learn the internals of context parallelism, you can skip it and start building 🚀

We're going to be using word `shard` extensively in the following sections, so let's define it first. If we call tensor `sharded` across `Dth` dimension, across `N` ranks, we mean that this tensor is split into `N` parts, where each part of the tensor has shape `[..., D//N, ...]`.


## So how does it work?

Context parallelism works on sharding the `Q, K and V` matrices across the sequence dimension. Each rank has its assigned shard of `Q`, let's call it `Q_i`. This matrix stays only on this rank, during the whole computation. Similarly, each rank has its own shard of `K` and `V`, let's call them `K_i` and `V_i`. Then, each rank calculates attention with its own shard of `Q_i`, `K_i` and `V_i`, let's call it `attn_i`. During this computation, a communication kernel is launched to gather the `Ks` and `Vs` from all other ranks. What communication primitive is used, depends on the `context_parallel_shard_rotation` option.
This way, each rank gets to calculate local attention, first with `Q_i`, `K_i` and `V_i`, then with `K_j` and `V_j` from all other ranks. As each rank holds `Q, K and V` matrices that are sharded across the sequence dimension, the resulting matrices are smaller and can fit on a single GPU.

We can formalize this in a following pseudocode:
```python
comm_kernel = {"allgather": allgather, "alltoall": alltoall}[context_parallel_shard_rotation]
Qi, Ki, Vi = shard(Q, K, V, seq_dim)
attn[i] = attn(Qi, Ki, Vi)
for j in range(context_parallel_size):
    Kj, Vj = comm_kernel()
    attn[j] = attn(Qi, Kj, Vj) # [batch, num_heads, seq_len // context_parallel_size, head_dim]

final_attn = combine(attn)
```

## all-to-all vs all-gather

### all-gather
So what's the difference between all-to-all and all-gather? With all-gather, the communication is very simple. After (well, before, as it usually takes longer) we compute the local attention `attn_i` we launch an all-gather to gather all other `Ks` and `Vs` from all other ranks. As this communication is done, each rank has all the `Ks` and `Vs` from all other ranks, and can compute the attention with them sequentially.
In ideal scenario, all-gather finishes in the exact moment as the calculation of `attn_i` is done. However, this never happens in practice, so the ideal real overlap is achieved when the full `attn_i` is overlapped with a part of the communication, then to start the computation with `K_j` and `V_j`, we wait for the all-gather to finish.

### all-to-all
All-to-all, or sometimes called `ring-rotation` utilizes a ring-like communication pattern. After concluding `attn_i` computation, an all-to-all is launched to send `K_i` and `V_i` to the neighbouring ranks. We then repeat this `context_parallel_size-1` times, so that each rank sees all the shards of `K` and `V` from all other ranks once. In ideal scenario, we prefetch shards `K_i+1` and `V_i+1` from the neighbouring rank and this communication is exactly overlapped with computation of our current `attn_i`. Again, realistically, this perfect overlap doesn't ever happen. Given the nature of this approach, if we don't achieve perfect overlap, the penalty is way larger than with all-gather.

## How to choose the right rotation method?
In theory, all-to-all should be the better choice. Though in practice, it rarely is. Therefore, we default to all-gather, as it's more likely to achieve better performance. Extensive [benchmarks](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082) from the `torchtitan` team also shows that all-to-all rarely outperforms all-gather. Though, we still provide both options, as you might find one to be better for your use case.

You can directly see this issue in the profiler output in the image below:
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/cp_all_to_all.png" alt="all-to-all profiler output" />
  <br>
  <em>Figure 1: In red you can see the idle time, while we wait for the all-to-all kernel to finish. It's also highlighted in the first blue bar, where we can see it takes ~250us to finish, which is repeated N-1 times for each attention call, where N is the context parallel size.</em>
</p>


## Why only FSDP2?

We only support context parallelism with `FSDP2` for now, as we create a joint mesh of `context_parallel_size` and `dp_shard_size` to 
utilize its full potential. In the profiler output in the image below, you can see that the computation of `attn_i`

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/cp_why_fsdp2.png" alt="why FSDP2+CP" />
  <br>
  <em>Figure 2: In blue rectangles (Stream 23), you can see that the pre-fetch of `FSDP` shard is fully overlapped with the computation of attention (Stream 7), while in red rectangles (Stream 24), you can see that the all-gather kernel results in a bubble of idle time, in which our compute stream (7) is idle.</em>
</p>

In the figure above, you can also note the difference between all-to-all and all-gather. While in all-to-all (Figure 1), we launch a communication kernel N-1 times for each attention call, in all-gather (Figure 2), we launch a communication kernel only once. This results in a bigger bubble, but it only happens once per attention call, while in all-to-all, it happens N-1 times.
