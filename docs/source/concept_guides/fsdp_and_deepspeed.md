<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Moving between FSDP And DeepSpeed (DRAFT)

🤗 Accelerate offers flexibilty of training frameworks, by integrating two extremely powerful tools for distributed training, namely [Pytorch FSDP](../usage_guides/fsdp.md) and [Microsoft DeepSpeed](../usage_guides/deepspeed.md). The aim fo this article is to draw parallels, as well as to outline potential differences, to empower the user to switch seamlessly between these two frameworks.

<Tip>
  To switch between the frameworks, we recommend 🤗 `accelerate launch`.

  Simply pass [FSDP and DeepSpeed arguments](../package_reference/cli#accelerate-launch) directly to `accelerate launch` or pass `--config_file`. No need for any code instrumentation!

  Exemplar 🤗 Accelerate configurations can be found here for [DeepSpeed](../usage_guides/deepspeed#accelerate-deepspeed-plugin) and [FSDP](../usage_guides/fsdp#how-it-works-out-of-the-box). 
 
</Tip>

This article is written with 🤗 Accelerate in mind, for single-node, multi-GPU, scenarios only. No TPU aspects will be considered.

## Configuring Functionalities

Model tensors are split into different GPUs in attempt scale up model sizes; this is termed *sharding* in FSDP, and *partitioning* in DeepSpeed. FSDP sharding and DeepSpeed ZeRO (partitioning) stages, are configured by `--fsdp_sharding_strategy`, and `--zero_stage`, respectively.  In particular, FSDP `FULL_SHARD` maps to DeepSpeed ZeRO stage `3`; see this [comprehensive mapping between FSDP sharding and DeepSpeed ZeRO settings](../usage_guides/fsdp#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages). The below table summarizes and groups similar settings:

Group | Framework | Configuration | Example | Restrictions (if any)
--|--|--|--|--
sharding / partitioning | FSDP<br>DeepSpeed | `--fsdp_sharding_strategy`<br>`--zero_stage` | `1` (`FULL_SHARD`) <br>`3` | 
offload | FSDP<br>DeepSpeed | `--fsdp_offload_params`<br>`--offload_param_device`<br>`--offload_optimizer_device` | `true`<br>`cpu`<br>`cpu` | all or nothing <br><br> 
model loading | FSDP<br>DeepSpeed | <span style="white-space:nowrap;">`--fsdp_cpu_ram_efficient_loading`</span><br>`--zero3_init_flag` | `true`<br>`true` | <br>only ZeRO 3
efficient checkpointing | FSDP<br>DeepSpeed | `--fsdp_state_dict_type`<br>`--zero3_save_16bit_model` |  `SHARDED_STATE_DICT`<br>`true` |  <br>only ZeRO 3
pipeline | FSDP<br><br>DeepSpeed | `--fsdp_forward_prefetch`<br>`--fsdp_backward_prefetch`<br>None | `true`<br>`BACKWARD_PRE` | <br><br>
model | FSDP<br><br>DeepSpeed |  `--fsdp_auto_wrap_policy`<br><span style="white-space:nowrap;">`--fsdp_transformer_layer_cls_to_wrap`</span><br>None | `TRANSFORMER_BASED_WRAP`<br><Layer Class> |<br>Usually not needed <br>Transparent to user.
parameters summoning | FSDP<br>DeepSpeed | `--fsdp_use_orig_params`<br>None | `true` | required for `torch.compile`<br>Transparent to user
parameters syncing | FSDP<br>DeepSpeed | `--fsdp_sync_module_states`<br>None | `true` | 
training | FSDP<br>DeepSpeed | None<br>`--gradient_accumulation_steps`<br>`--gradient_clipping` | <br>`auto`<br>`auto` | Transparent to user

For detailed descriptions of the above, refer to [🤗 `Accelerate` launch documentation](../package_reference/cli#accelerate-launch).

<Tip>

    To access other DeepSpeed configurations, such as mixed precision settings, 
    you need to link a `deepspeed_config_file`, see the [documentation](../usage_guides/deepspeed#deepspeed-config-file).  
    
</Tip>

**Checkpointing**

Do note that while FSDP can be configured via `--fsdp_state_dict_type` to save either full / sharded checkpoints, but DeepSpeed [only saves sharded checkpoints](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#saving-training-checkpoints).

<Tip>

    For DeepSpeed Zero3, it is recommended to also pass a `--zero3_save_16bit_model true` in the [`Accelerate` launch arguments](../package_reference/cli#accelerate-launch) as this is typically much more efficient.

</Tip>

**Offloading**

FSDP only allows *all-or-nothing* offload, but DeepSpeed can offload parameters and optimizer differently. Furthermore, DeepSpeed also supports [offloading to NVME](https://www.deepspeed.ai/docs/config-json/#parameter-offloading).

**Prefetching**

FSDP allows two prefetching configurations `--fsdp_forward_prefetch` and `--fsdp_backward_prefetch` to improve overlap of comms / computation at a cost of extra memory, see [FSDP documentation](https://pytorch.org/docs/stable/fsdp.html). 
For DeepSpeed, the prefetching is always on, and only certain hyperparams like `stage3_prefetch_bucket_size` [can be configured for Zero3](https://www.deepspeed.ai/docs/config-json/#parameter-offloading); 🤗 [`accelerate`] will set these hyperparams automatically.

<Tip>

    For FSDP set `fsdp_backward_prefetch: BACKWARD_PRE` for improved throughputs if memory allows.

</Tip>

**Model Loading**

While FSDP require an explicit `--fsdp_cpu_ram_efficient_loading true` to activate efficient model loading, 🤗 `transformers` will activate the similar feature whenever DeepSpeed Zero3 is used.

<Tip>

    For FSDP, whenever setting `--fsdp_cpu_ram_efficient_loading true`, please also set `--fsdp_sync_module_states true`, otherwise the model will not load properly. 

</Tip>

**Model**

FSDP requires an explicit `--fsdp_auto_wrap_policy` for the algorithm to decide how to schedule the all-gather and reduce-scatter operations. But for DeepSpeed this is transparent to the user.

<Tip>

    For FSDP, simply set `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`. Do not set `fsdp_transformer_layer_cls_to_wrap` if using the latest `transformers` versions.

</Tip>

**Parameters Summoning**

FSDP requires an explicit `--fsdp_use_orig_params` flag if using `torch.compile`, see [the pytorch documenation](https://pytorch.org/docs/stable/fsdp.html#module-torch.distributed.fsdp). For DeepSpeed this is transparent to the user.

<Tip>

    For FSDP, when using `torch.compile` please set `fsdp_use_orig_params: True`.

</Tip>


**Training**

Deepspeed requires explicit `--gradient_accumulation_steps` and `--gradient_clipping` flags. For FSDP this is transparent to the user.

<Tip>

    When using DeepSpeed, set `gradient_accumulation_steps: "auto"` and `gradient_clipping: "auto"` to automatically pick up values set in [`TrainingArguments`].

</Tip>


## On Differences in Data Precision Handling

To discuss the how data precision is handled in both FSDP and Deepspeed, it is instructive to first give a flow of how model parameters are handled in these frameworks. Before the model / optimizer parameters are distributed across GPUs, parameter preparation is involved to first "flatten" them to  one-dimensional [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch-tensor)'s. The implementation of FSDP / DeepSpeed varies in the respect of the `dtype` in which these "flattened" parameters are stored, and there are ramifications with regards to how [`torch.Optimizer`](https://pytorch.org/docs/stable/optim.html#module-torch.optim)'s allocate their `dtypes`'s. The table below outlines the processes for both frameworks; the "Local" column indicates the process occurring at a per-gpu level, therefore any memory overheads by upcasting should be understood to be amortized by the number of gpus used.

Process | Local | Framework | Details
--|--|--|--
Loading, i.e., [`AutoModel.from_pretrained(..., torch_dtype)`] |  
Preparation, i.e., creation of "flat params" | ✅ | FSDP<br>DeepSpeed | created in `torch_dtype`.<br> disregards `torch_dtype`, created in `float32`.
Optimizer initialization | ✅ | FSDP<br>DeepSpeed  | creates parameters in `torch_dtype`<br> creates parameters in `float32`
Training Step, i.e, forward, backward, reduction | | FSDP<br>DeepSpeed  | follows [`MixedPrecision`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)<br> follows `deepspeed_config_file` mixed precision settings.
Optimizer (Pre-Step) | ✅ | FSDP<br>DeepSpeed | upcasting (if any) to `torch_dtype`<br>upcasted to `float32`
Optimizer (Actual Step) | ✅ | FSDP<br>DeepSpeed  | occurs in `torch_dtype` <br> occurs in `float32`.

<Tip warning={true}>

    Therefore when using DeepSpeed a small number of GPUs, be aware of potentially significant memory overheads due to the upcasting during preperation.

</Tip>

<Tip warning={true}>

    With FSDP, in the absence of mixed precision, it is possible to operate the [`torch.Optimizer`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) in low precision `torch_dtype`, which may be helpful when using small number of GPUs. 

    On the other hand with mixed precision, then FSDP (like DeepSpeed) will upcast in the model preperation step (c.f. table above).

</Tip>


To clarify the above table consider the concrete examples below; the optimizer pre- and actual step combined for brevity. With FSDP it is possible to operate in the two modes shown below, but DeepSpeed can only operate in one.

Framework | Model Loading (`torch_dtype`) | Mixed Precision | Preparation (Local) | Training | Optimizer (Local)
--|--|--|--|--|--
FSDP | bf16 | default (none) | bf16 | bf16 | bf16
FSDP | bf16 | bf16 | fp32 | bf16 | fp32
DeepSpeed   | fp32 | bf16 | fp32 | bf16 | fp32