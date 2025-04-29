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

# FSDP1 vs FSDP2

This guide explains the key differences between `FSDP1` and `FSDP2` and helps you migrate your existing code to use `FSDP2` with minimal changes.

## How is FSDP2 better than FSDP1?

First, we want to understand how `FSDP1` and `FSDP2` work internally to understand the differences between them. This also helps us understand the limitations of `FSDP1` and how `FSDP2` solves them.

We'll be discussing a scenario where we have a single `Layer` that contains 3 `Linear` layers and is wrapped using `FSDP` to be sharded across 2 GPUs.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/layer.png" alt="Layer">
</div>

### FSDP1
First, we have to understand the original `FSDP1` and the limitations it brings. It represents each `FSDP` module as a single `FlatParameter` which is a single 1D tensor that contains all of the module parameters, which then get sharded across ranks. I.e. if you wrap the `Layer` with `FSDP1`, you'd achieve something as such:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp1.png" alt="FSDP1">
</div>

You might notice a problem. The whole `Layer` gets flattened into a single `FlatParameter`, which then gets sharded across ranks. But if it's a single `FlatParameter` object, how do we store metadata? That is one of the limitations. Properly storing per-parameter metadata such as `dtype`, `requires_grad`, etc. is not possible without some ugly hacks.

### FSDP2
This is why `FSDP2` was introduced. It doesn't use `FlatParameter`, instead it uses `DTensor` which is short for "Distributed Tensor". Each `DTensor` basically represents a vanilla `torch.Tensor` that has been sharded across ranks. It contains metadata about the original `torch.Tensor` and how it's sharded, what is the [placement type](https://pytorch.org/docs/stable/distributed.tensor.html#module-torch.distributed.tensor.placement_types) and so on. This is why it's called `per-parameter sharding`. The following figure shows the difference:

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/fsdp2.png" alt="FSDP2">
</div>

Each Parameter of the original `Layer` is sharded across the 0th dimension, and split between 2 GPUs. Now, each `Linear` layer is a separate `DTensor` and storing metadata per-parameter is possible and straightforward.


> [!TIP] 
> In the image above, the tensors were sharded across the 1st dimension for the sake of fitting the image on the screen, in reality, they are sharded across the 0th dimension as stated above

## What does FSDP2 offer?

`FSDP2` is a new and improved version of PyTorch's fully-sharded data parallel training API. Its main advantage is using `DTensor` to represent sharded parameters. Compared to `FSDP1`, it offers:
- Simpler internal implementation, where each `Parameter` is a separate `DTensor`
- Enables simple partial parameter freezing because of the above, which makes methods as [`LORA`](https://arxiv.org/abs/2106.09685) work out of the box
- With `DTensor`, `FSDP2` supports mixing `fp8` and other parameter types in the same model out of the box
- Faster and simpler checkpointing without extra communication across ranks using `SHARDED_STATE_DICT` and [`torch.distributed.checkpoint`](https://pytorch.org/docs/stable/distributed.checkpoint.html), this way, each rank only saves its own shard and corresponding metadata
- For loading, it uses a `state_dict` of the sharded model to directly load the sharded parameters
- Support for asynchronous checkpointing, where parameters are first copied to CPU memory, after this, main thread continues training while another thread stores the parameters on disk
- Memory efficiency and deterministic memory usage, `FSDP2` doesn't use `recordStream` anymore and uses stream-to-stream synchronization (for more technical details see [this forum post](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486) and [this issue](https://github.com/pytorch/pytorch/issues/114299))
- In the future, optimizations of the communication patterns via `torch.compile` are planned, further improving the performance and memory efficiency


## API Differences

We have already discussed the internal differences, now let's discuss the differences, you, as a user, will need to know. 

Here are the main changes in configuration options when using `FSDP2` through the `accelerate` CLI:

Previous (`FSDP1`) | New (`FSDP2`) | What Changed
-- | -- | --
`--fsdp_sharding_strategy` | `--fsdp_reshard_after_forward` | replaces `--fsdp_sharding_strategy`, changed to `true` (previously `FULL_SHARD`) or `false` (previously `SHARD_GRAD_OP`)
`--fsdp_backward_prefetch` | \*\***REMOVED**\*\* | `FSDP2` uses previous `BACKWARD_PRE` option by default, as only this allows communication and computation overlap
`--fsdp_forward_prefetch` | \*\***NOT YET IMPLEMENTED**\*\* | How to implement this is under active discussion, for now it is not supported in `FSDP2`
`--fsdp_sync_module_states` | \*\***REMOVED**\*\* | with `FSDP2`, this parameter becomes redundant
`--fsdp_cpu_ram_efficient_loading` | `--fsdp_cpu_ram_efficient_loading` | if `true`, `FSDP2` will similarly load the model only on rank 0, and then parameters get synced to other ranks, this is the same behavior as `FSDP1`, however, setting `--fsdp_sync_module_states` isn't required anymore
`--fsdp_state_dict_type` | `--fsdp_state_dict_type` | `LOCAL_STATE_DICT` becomes obsolete and with `FSDP2` `SHARDED_STATE_DICT` is the default option, which results in no extra communication and each rank saving its own shard, other possible option is `FULL_STATE_DICT` which results in extra communication and spike in memory usage but saves the full model from rank 0.
`--fsdp_use_orig_params` | \*\***REMOVED**\*\* | `FSDP2` uses a `DTensor` class on the background, which means it *always* uses the original parameters by default
\*\***NEW**\*\* | `--fsdp_version` | `1` is the default option, to not break existing code, set to `2` to use `FSDP2`

For all other options that remain unchanged, see the [`FSDP` documentation](../usage_guides/fsdp.md).

## How to Switch to FSDP2

### If using Python code:
Simply set `fsdp_version=2` when creating your plugin and replace options according to the table above.

```python
from accelerate import FullyShardedDataParallelPlugin, Accelerator

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2
    # other options...
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

### If using YAML config:
Use our conversion tool:
```bash
accelerate to-fsdp2 --config_file config.yaml --output_file new_config.yaml
```

This will automatically convert all FSDP1 settings to their FSDP2 equivalents. Use `--overwrite` to update the existing file instead of creating a new one.
