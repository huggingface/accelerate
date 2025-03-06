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

This guide will cover the key differences between `FSDP1` and `FSDP2`. It is also going to help you change your existing code to use `FSDP2` with minimal changes.


## What is FSDP2?

`FSDP2` is a new API for fully-sharded data parallel training in PyTorch. It is a successor to [`FSDP1`](../usage_guides/fsdp.md) and it is supposed to simplify
the internal implementation of the API, while also providing extensions to enable flexible freezing of model parameters, gathering of `fp8` parameters, faster and 
simpler checkpointing and more.

## Key Differences

The following table summarizes the key differences between `FSDP1` and `FSDP2` configuration options configurable through the CLI:

Configuration `FSDP1` | Configuration `FSDP2` | Changed Behaviour
-- | -- | --
**NEW** | `--fsdp_reshard_after_forward` | replaces `--fsdp_sharding_strategy`, changed to `true` (previous `FULL_SHARD`) or `false` (previous `SHARD_GRAD_OP`)
`--fsdp_sharding_strategy` | **DEPRECATED** | `FSDP2` uses the `reshard_after_forward` option only, this gets ignored if `FSDP2` is used
`--fsdp_backward_prefetch` | **REMOVED** | `FSDP2` uses previous `BACKWARD_PRE` option by default, as only this allows communication/computation overlap
`--fsdp_state_dict_type` | **REMOVED** | `FSDP2` always uses `SHARDED_STATE_DICT`, i.e. each rank only checkpoints the shard of the model on it, resulting in no extra communication
`--fsdp_forward_prefetch` | **NOT YET IMPLEMENTED** | How to implement this is under active discussion, for now it is not supported
`--fsdp_cpu_ram_efficient_loading` | **TODO** | **TODO**
`--fsdp_sync_module_states` | **TODO** | **TODO**
`--fsdp_use_orig_params` | **REMOVED** | `FSDP2` uses a `DTensor` class on the background, which means it *always* uses the original parameters by default
**NEW** | `--fsdp_version` | `FSDP2` is the default, `FSDP1` can be selected by setting this to `1`

These are the configuration options that have changed in `FSDP2`, to read more about the options that didn't change, you can read the documentation for [`FSDP`](../usage_guides/fsdp.md).

`FSDP2` also supports the original `FullyShardedDataParallelPlugin` interface, so you can still use it to override the options not exposed in the CLI, only by setting the `fsdp_version` to `2`.

```py
from accelerate import FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2
    ...
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

## How to switch to FSDP2?

If you're using the `FullyShardedDataParallelPlugin` interface, you can just replace the `
#TODO:


