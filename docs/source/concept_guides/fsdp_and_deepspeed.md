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

# FSDP VS DeepSpeed [DRAFT]

🤗 Accelerate integrates two extremely powerful tools for distributed training, namely [Pytorch FSDP](../usage_guides/fsdp.md) and [Microsoft DeepSpeed](../usage_guides/deepspeed.md). It is important to note similarities / differences in order to make an informed choice which framework that will better suit the desired use case.

<Tip>
  To switch betwen the frameworks, we recommend 🤗 `accelerate launch`.

  Simply pass [FSDP and DeepSpeed arguments](../package_reference/cli.md#accelerate-launch) directly to `accelerate launch` or pass `--config_file`. No need for any code instrumentation!

  For examplar configurations see [DeepSpeed Config](../usage_guides/deepspeed#accelerate-deepspeed-plugin). and [FSDP Configuration](../usage_guides/fsdp.md#how-it-works-out-of-the-box). 
  

</Tip>

<!--
The aim of this concept guide is to elucidate similarities/differences with empirical observations and code aspects.  We assume that 🤗 Accelerate is used and configured using the [FSDP and DeepSpeed arguments](../package_reference/cli.md#accelerate-launch). No TPU aspects are discussed. Also we focus only on single-node aspects.
-->

This article is written with 🤗 Accelerate in mind, for single-node GPU scenarios only. No TPU aspects will be considered.

## Configuring Various Functionalities

Training parameters are seperated into different GPUs to scale up for large models; this is termed *sharding* in FSDP, and *partioning* in DeepSpeed. Since the terminologies used in FSDP and DeepSpeed are disparate, there are various guides that [reconcile the mapping between FSDP sharding and DeepSpeed ZeRO](../usage_guides/fsdp.md#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages). This section aims to reconcile these differences for a 🤗 Accelerate user that desires to use both frameworks in an equivalent manner.

In general, DeepSpeed offers more fine-grained control, but may incur extra memory consumption as model and optimizer parameters are always upcasted to `float32`. 
Below shows a mapping between FSDP and DeepSpeed configuration. 
Note that to access advanced DeepSpeed configs via `accelerate launch` beyond
[those exposed by accelerate](../package_reference/cli.md#accelerate-launch), one can link a full DeepSpeed config and pointing to it using the accelerate `deepspeed_config_file`. 
 
Configuration | FSDP | DeepSpeed
--|--|--
sharding/partitioning | `--fsdp_sharding_strategy` | `--zero_stage`
offload | `--fsdp_offload_params` | `--offload_optimizer_device, --offload_param_device`
efficient weights loading | `--fsdp_cpu_ram_efficient_loading` | `--zero3_init_flag`
checkpointing | `--fsdp_state_dict_type` | `--zero3_save_16bit_model`
pipeline | `--fsdp_backward_prefetch, --fsdp_backward_prefetch` | 
model | `--fsdp_auto_wrap_policy, --fsdp_transformer_layer_cls_to_wrap` | 
parameters summoning | `--fsdp_use_orig_params` | 
parameters syncing | `--fsdp_sync_module_states` | 
training |  | `--gradient_accumulation_steps, --gradient_clipping`

<Tip>
    When training transformers, there are some FSDP configuration recommendations that should be followed.

    - set `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`, and `fsdp_transformer_layer_cls_to_wrap` not needed for latest `transformers` versions.
    - when setting `fsdp_cpu_ram_efficient_loading: True`, ensure `fsdp_sync_module_states: True` so that parameters are communicated from main process.
    - is using `torch.compile` set `fsdp_use_orig_params: True`

</Tip>

<Tip>
    These are also the DeepSpeed configuration recommendations that should be followed.

    - set `gradient_accumulation_steps: "auto"` and `gradient_clipping: "auto"` to automatically pick up values set in [`TrainingArguments`].

</Tip>

## Differences in Training Precisions

Both FSDP and DeepSpeed have logic for sharding/partitioning the model / optimizer parameters across GPUs; however there are some differences to be aware of. Both of these
frameworks will create new [`torch.Tensor`]'s to hold "flattened" parameters, which will be used to instantiate the optimizers in each shard/partition. However since the [`torch.Optimizer`]'s allocate their `dtypes`'s based on the optimized parameters, it is important to note the following:

<Tip warning={true}>

    FSDP will retain the model's `dtype`; if a model is loaded in `float16`, then that will be the `dtype` of the flattened model parameters, and also that of the optimizer, etc. This will be discussed more clearly below. 
    
    DeepSpeed however will always upcast the parameters to `float32`, so even if the model was loaded in `float16`, the optimizer will see `float32` parameters. In other words, DeepSpeed only operates in mixed precision. There is no reason to load in low precision and yet train with DeepSpeed.

    This difference may result in different observations during training.

</Tip>

For FSDP there are two options i) load and train/optimize the model in low precision, or ii) load in full precision and configure [`MixedPrecision`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision) to have activations / reduction performed in low precision. But for DeepSpeed always load the model in full precision. To summarize:

Framework | Load Model | Mixed Precision | Flat Params | Forward / Backward | Optimizer
--|--|--|--|--|--
FSDP | bf16 | None | bf16 | bf16 | bf16
FSDP | bf16| fp32 | fp32 | bf16 | fp32
DeepSpeed   | bf16 | bf16 | fp32 | bf16 | fp32