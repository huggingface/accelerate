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

With the advent of large language models, and recently reasoning models, the sequence length has been growing rapidly. This, combined with quadratic memory complexity of attention, has led to a need for more efficient ways to train models with long sequences.
With sequence length of 128k, the memory requirement of the attention matrix is `128k * 128k * 2 bytes * num_heads = ~32 GB * num_heads` for `bf16` precision, given vanilla attention implementation. Granted, with usage of `flash attention` or `SDPA` which do not materialize these attention weights, this decreases drastically, but the growth in memory requirements is still considerable.

Context parallelism allows us to shard the inputs to the attention computation along the sequence dimension and compute the attention in parallel on multiple GPUs. With this, we can train models with long sequences, scaling potentially to 1M+ sequence length.

## Supported backends

Multiple backends are currently supported

1. `torch`: PyTorch/FSDP2,which implements several of Ring Attention context parallel protocols [tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) and [api](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel).
2. `deepspeed`: DeepSpeed/ALST/UlyssesSP, which implements sequence parallelism using attention head parallelism: [tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/) and [paper](https://arxiv.org/abs/2506.13996)

## How to use context parallelism?

```diff
from accelerate.utils import ParallelismConfig, TorchContextParallelConfig

+ cp_config = TorchContextParallelConfig(
+       cp_comm_strategy="alltoall", # no need to use cp_config at all, if you want to use the default "allgather"
+ )

+ parallelism_config = ParallelismConfig(
+     cp_size=8,
+     cp_handler=cp_config,  # or just cp_size=8, if you want to use the default "allgather"
+ )

accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```

By default the `torch` backend is selected, but you can select the deepspeed backend via:

```python
parallelism_config = ParallelismConfig(
    backend="deepspeed",
    cp_size=4,
    cp_handler=DeepSpeedContextParallelConfig(
        seq_length=256,
        attn_implementation="sdpa"
    ),
)
```
See the following sections for nuances of each backend.

As with any other feature in 🤗`accelerate`, you can enable context parallelism also by passing the corresponding flags to `accelerate launch`. In this case, it's no different:

```bash
accelerate launch --parallelism-config-cp-size 8 --parallelism-config-cp-comm-strategy [allgather|alltoall] ...
```

> [!Tip]
> You can also set the `cp_size` and `cp_comm_strategy` in the `accelerate config` command, which will save them in your `accelerate` configuration file, so you don't have to pass them every time you launch your script.

> [!Tip]
> Context parallelism is compatible with other parallelism strategies, such as data parallelism, tensor parallelism and FSDP2.
> You can simply combine them by setting your parallelism sizes to the desired values, e.g. `--parallelism-config-dp-size 8 --parallelism-config-tp-size 2 --parallelism-config-cp-size 8`. Or you can use the `ParallelismConfig` class to set them programmatically.

## Torch/FSDP2 backend

> [!Warning]
> Context parallelism is tightly coupled  with `FSDP2`, which you can learn more about in the [FSDP2 introduction](fsdp1_vs_fsdp2.md). Meaning, context parallelism only works if you use `FullyShardedDataParallelPlugin` or `--use-fsdp` with version set to 2 to your
> program. If no `FSDP2` is used, error will be raised.

> [!Warning]
> `torch`-backend Context parallelism works only with [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) and only with no mask or causal mask. We can't properly detect this for you, so it's your responsibility to ensure that you are using `SDPA` with no mask or causal mask. If you use any other attention implementation, it will raise an error.

After enabling context parallelism with the methods mentioned above, you can then apply it to your training loop. We provide a thin wrapper around [`torch.distributed.tensor.experimental.context_parallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel) that you can use in your training loop, that abstracts some of the complexity of using it (more on this later). To minimize the changes you have to do in your training loop, we provide a context manager that is a `noop` if context parallelism is not enabled, and applies the context parallelism if it is enabled. This way, you can use it in your training loop without changing any code based on your parallelism configuration.
You can use it as follows:

```python
for batch in dataloader:
    with accelerator.maybe_context_parallel(
        buffers=[batch["input_ids"], batch["attention_mask"]],
        buffer_seq_dims=[1, 1],
        no_restore_buffers={batch["input_ids"], batch["labels"]},
    ):
        outputs = model(**batch)
        ...
```

> [!Warning]
> This context manager has to be recreated with each training step, as shown in the example above. It's crucial to do so.

This can scale your context size to 1M+ sequence length potentially. Below, we showcase speed and memory usage of context parallelism for up-to 256k context size. We can see that when we double the context size and number of GPUs, we can achieve consistent memory usage, potentially enabling endless context length scaling.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/cp_perf.png" alt="context parallelism memory usage" />
  <br>
  <em>Figure 1: Memory usage and speed of context parallelism for up-to 256k context size.</em>
</p>

> [!Tip]
> These examples were created with a script you can find [in the examples folder](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/nd_parallel.py). To run the example on 8 H100 GPUs (128k sequence length), you can use the following command:
> ```bash
> accelerate launch --use-fsdp --fsdp-activation-checkpointing=TRUE examples/torch_native_parallelism/nd_parallel.py --cp-size=8 --sequence-length=128000
> ```

## DeepSpeed/ALST/UlyssesSP backend

ALST/UlyssesSP implements a sequence parallelism using attention head parallelism as explained in [this paper](https://arxiv.org/abs/2506.13996) - for simplicity we re-use the concept and the setup of context parallelism, which from the user's end of view is the same - multiple gpus are used to process a single batch.

To give a sense of what ALST made possible - it allowed us to train in bf16 with 500K tokens on a single H100 GPU, 3.7M on a single node, and 15M on Llama-8B using just four nodes. This feature of HF Accelerate enables only 1 of the 3 ALST components so the achievable sequence length will be smaller. You'd want TiledMLP, Activation checkpoint offload to CPU and a few other things enabled to get the full power of ALST, for details please refer to [this tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/).

To configure the `deepspeed` backend:

```python
parallelism_config = ParallelismConfig(
    backend="deepspeed",
    cp_size=4,
    cp_handler=DeepSpeedContextParallelConfig(
        seq_length=256,
        seq_length_is_variable=True,
        attn_implementation="sdpa",
    ),
)
accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```

- `cp_size` is the degree of the sequence parallelism - in the above example it's 4, therefore 4 gpus will be used to process a single batch.
- `seq_length` and `seq_length_is_variable` are used to deal with sequence lengths. If `seq_length_is_variable=True` the backend will work with a sequence length that may change between batches, in which case `seq_length` value can be set to anything divisible by the context parallel degree or not set at all. In this case on every `forward` the sequence variables will be derived from input. If `False` then `seq_length` needs to match the batch's sequence length dimension, which then will have to be padded to be always the same. The default is `True`.
- `attn_implementation` is one of `sdpa`, `flash_attention_2` or ``flash_attention_3`. This sequence parallel implementation uses `position_ids` instead of `attention_mask` therefore `eager` can't work here until it'd support working with `position_ids`. Also please note that `sdpa` doesn't handle correctly combined into one multiple-samples, it'd attend to the whole sample as one. If the samples aren't combined `sdpa` will work correctly. Therefore Flash Attention should be the ideal choise as it always works.

Instead of setting these values in `DeepSpeedContextParallelConfig` object, you can also use the environment variables to accomplish the same - here they are correspondingly to the end of the list above.
- `PARALLELISM_CONFIG_CP_SEQ_LENGTH`
- `PARALLELISM_CONFIG_CP_SEQ_LENGTH_IS_VARIABLE`
- `PARALLELISM_CONFIG_CP_ATTN_IMPLEMENTATION`

If not passed in the code `cp_size` can be set via `--parallelism_config_cp_size` CLI argument.

Please note that a lot of magic is hidden inside [UlyssesSPDataLoaderAdapter](https://github.com/deepspeedai/DeepSpeed/blob/64c0052fa08438b4ecf4cae30af15091a92d2108/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L442). It's used behind the scenes, wrapping your original DataLoader object, but you should be aware of it should you run into any problems. It also automatically injects the correct `shift_labels` into the batch dictionary, before the batch gets sharded across the participating ranks.

Now the only remaining piece to start using ALST/UlyssesSP is to aggregate the loss across ranks using a differentiable `all_gather` to get the grads right. The following code does it, while also exlcuding any masked out with `-100` tokens, to get the correct average:

```python
cp_size = parallelism_config.cp_size if parallelism_config else 1
if cp_size > 1:
    sp_group = accelerator.torch_device_mesh["cp"].get_group()
    sp_world_size = parallelism_config.cp_size

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

    if cp_size > 1:
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

If you use [Liger Kernel](https://github.com/linkedin/Liger-Kernel) it already knows how to handle `shift_labels` so you don't need to go through manual loss calculation, just calling `model(**batch)` will already get the `loss` calculated and done in a very memory-efficient way. If you didn't know about Liger-Kernel - it's highly recommended to be used especially for long sequence length since it liberates a lot of working memory that can be used for handling longer sequences.

If you want to see what HF Accelerate did behind the scenes please read [this full integration tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/).

For an example of an Accelerate training loop with enabled ALST/UlyssesSP see [examples/alst_ulysses_sequence_parallelism](https://github.com/huggingface/accelerate/blob/main/examples/alst_ulysses_sequence_parallelism).

[!Warning]
> This API is quite new and still in its experimental stage. While we strive to provide a stable API, it's possible some small parts of the public API will change in the future.


## Accelerate's interface

The context manager takes a few arguments, that are used to configure the context parallelism.

- `buffers`: This is a list of tensors that are to be sharded across the sequence dimension. These tensors are usually input ids, labels and attention mask.
- `buffer_seq_dims`: This is a list of integers, that specify the sequence dimension of the buffers, in the order of the `buffers` list. If you pass `buffers=[input_ids, shift_labels]` with both having shape `[batch_size, sequence_length]`, you would pass `buffer_seq_dims=[1, 1]`.
                     as the sequence dimension is the second dimension of the tensors. This is required for correct computation of the model outputs.
- `no_restore_buffers`: The implementation of context parallelism modifies the buffers in-place, converting them to `torch.distributed.tensor.Dtensor`s. After the context manager exits, a communication kernel would need to be launched to restore the buffers to their original state (usually all-gather). This takes some time, so it is recommended to pass the same tensors as in the `buffers` argument, to avoid unnecessary communication, unless you are sure that you need to use the buffers after the context manager exits.


> [!Warning]
> Context parallelism is not compatible with `labels` that are a copy of `input_ids`, which models from 🤗 transformers can shift to enable causal language modeling themselves.
> Imagine this case:
> labels = [l1, l2, l3, l4, ... li]
> if we apply context parallelism, each rank would end up with a part of labels, such as this:
> labels_rank_0 = [l1, l2], labels_rank_1 = [l3, l4], ...
> after transformers modelling code shifts the labels, it would end up with:
> labels_rank_0 = [l2, PAD], labels_rank_1 = [l3, PAD], ...
> where `PAD` is a padding token. This would result in incorrect loss computation, as the labels are not aligned with the inputs anymore.
> Because of this, you need to manually shift the labels before passing them in the model


## Configurable options
Accelerate provides only a single option to configure context parallelism (except for `cp_size`)

- `cp_comm_strategy`: The rotation method to use for the shards. We strongly recommend keeping this as `"allgather"`, as it's very likely it will outperform `"alltoall"` in most cases.

Context parallel size is rather self-explanatory, it's the number of ranks across which the inputs are to be-sharded.
Context parallel shard rotation defines how the shards of the inputs are rotated across ranks. We'll cover the 2 options in more detail in the next section.

You can see an end-to-end example in the [ND parallel example](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py) file, where you can train an 8B model with up-to 128k context length on a single 8xH100 node. Using multi-node training, you can scale this to 1M+ sequence length on multiple GPUs. You can also seamlessly combine it with other parallelism strategies to fit your needs.

## Technical details

> [!Tip]
> This section is fairly technical, so if you don't need to learn the internals of context parallelism, you can skip it and start building 🚀

We're going to be using word `shard` extensively in the following sections, so let's define it first. If we call tensor `sharded` across `Dth` dimension, across `N` ranks, we mean that this tensor is split into `N` parts, where each part of the tensor has shape `[..., D//N, ...]`.


## So how does it work?

Context parallelism works on sharding the `Q, K and V` matrices across the sequence dimension. Each rank has its assigned shard of `Q`, let's call it `Q_i`. This matrix stays only on this rank, during the whole computation. Similarly, each rank has its own shard of `K` and `V`, let's call them `K_i` and `V_i`. Then, each rank calculates attention with its own shard of `Q_i`, `K_i` and `V_i`, let's call it `attn_i`. During this computation, a communication kernel is launched to gather the `Ks` and `Vs` from all other ranks. What communication primitive is used, depends on the `context_parallel_shard_rotation` option.
This way, each rank gets to calculate local attention, first with `Q_i`, `K_i` and `V_i`, then with `K_j` and `V_j` from all other ranks. As each rank holds `Q, K and V` matrices that are sharded across the sequence dimension, the resulting matrices are smaller and can fit on a single GPU.

We can formalize this in the following pseudocode:
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
In theory, all-to-all should be the better choice. Though in practice, it rarely is. Therefore, we default to all-gather, as it's more likely to achieve better performance. Extensive [benchmarks](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082) from the `torchtitan` team also show that all-to-all rarely outperforms all-gather. Though, we still provide both options, as you might find one to be better for your use case.

You can directly see this issue in the profiler output in the image below:
<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/cp_all_to_all.png" alt="all-to-all profiler output" />
  <br>
  <em>Figure 1: In red you can see the idle time, while we wait for the all-to-all kernel to finish. Highlighted in the first blue bar, you can see that it takes ~250us to finish, which is repeated N-1 times for each attention call, where N is the context parallel size.</em>
</p>


## Why FSDP1 is not supported

We only support context parallelism with `FSDP2`, as we create a joint mesh of `context_parallel_size` and `dp_shard_size` to
utilize its full potential.
How it works is: we shard the model across the joint mesh of size `cp_size*dp_shard_size`, which maximizes the memory savings.
This is a "free lunch" of sorts, as `FSDP` communication is fully overlapped with the computation of attention, as shown in the images below.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/cp_why_fsdp2.png" alt="why FSDP2+CP" />
  <br>
  <em>Figure 2: In blue rectangles (Stream 23), you can see that the pre-fetch of `FSDP` shard is fully overlapped with the computation of attention (Stream 7), while in red rectangles (Stream 24), you can see that the all-gather kernel results in a bubble of idle time, in which our compute stream (7) is idle.</em>
</p>

In the figure above, you can also note the difference between all-to-all and all-gather. While in all-to-all (Figure 1), we launch a communication kernel N-1 times for each attention call, in all-gather (Figure 2), we launch a communication kernel only once. This results in a bigger bubble, but it only happens once per attention call, while in all-to-all, it happens N-1 times.

## Data dispatching in joint mesh

We make sure to dispatch the same batch of data to the whole `cp` subgroup, so that the results are correct. (Meaning each rank in `cp` subgroup gets the same batch of data.) However, we also dispatch different batches to each rank of `dp_shard` group.
Imagine it like this:
```
# 8 GPUS, --dp_shard_size 4, --cp_size 2
# mesh = [[0, 1], [2, 3], [4, 5], [6, 7]]
# model is sharded across the whole mesh (each GPU holds 1/8 of the model)
# GPUs 0,1 = batch 0
# GPUs 2,3 = batch 1
... and so on.
```

