<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Low Precision Training Methods

The release of new kinds of hardware led to the emergence of new training paradigms that better utilize them. Currently, this is in the form of training
in 8-bit precision using packages such as [TransformersEngine](https://github.com/NVIDIA/TransformerEngine) (TE) or [MS-AMP](https://github.com/Azure/MS-AMP/tree/main).

For an introduction to the topics discussed today, we recommend reviewing the [low-precision usage guide](../usage_guides/low_precision_training.md) as this documentation will reference it regularly. 

## A Quick Chart

Below is a quick chart from the MS-AMP documentation showing the different bit-precisions for each solution during training:

Optimization Level | Computation(GEMM) | Comm | Weight | Master Weight | Weight Gradient | Optimizer States
-- | -- | -- | -- | -- | -- | --
FP16 AMP | FP16 | FP32 | FP32 | N/A | FP32 | FP32+FP32
Nvidia TE | FP8 | FP32 | FP32 | N/A | FP32 | FP32+FP32
MS-AMP O1 | FP8 | FP8 | FP16 | N/A | FP8 | FP32+FP32
MS-AMP O2 | FP8 | FP8 | FP16 | N/A | FP8 | FP8+FP16
MS-AMP O3 | FP8 | FP8 | FP8 | FP16 | FP8 | FP8+FP16

## `TransformersEngine`

`TransformersEngine` is the first solution to trying to train in 8-bit floating point. It works by using drop-in replacement layers for certain ones in a model that utilize their FP8-engine to reduce the number of bits (such as 32 to 8) without degrading the final accuracy of the model. 

Specifically, 🤗 Accelerate will find and replace the following layers with `TransformersEngine` versions:

* `nn.LayerNorm` for `te.LayerNorm`
* `nn.Linear` for `te.Linear`

As a result we wind up with a model that has most of its layers in BF16, while some layers are in FP8 reducing some of the memory. 

Anecdotally, we have noticed that performance gains don't really start showing when using `TransformerEngine` until a large majority of the layers
in the model are made up of those two layers to replace. As a result, only larger models have shown performance improvements when the number of parameters is around and upwards of a few billion. 

The `TransformerEngine` can receive many different arguments that customize how it performs FP8 calculations and what they do. A full list of the arguments is available below:

* `margin`: The margin to use for the gradient scaling.
* `interval`: The interval to use for how often the scaling factor is recomputed.
* `fp8_format``: The format to use for the FP8 recipe. Must be one of `E4M3` or `HYBRID`.
* `amax_history_len`: The length of the history to use for the scaling factor computation
* `amax_compute_algo`: The algorithm to use for the scaling factor computation. Must be one of `max` or `most_recent`.
* `override_linear_precision`: Whether or not to execute `fprop`, `dgrad`, and `wgrad` GEMMS in higher precision.

You can customize each of these as part of [`utils.FP8RecipeKwargs`] to help optimize performance of your models.

If we notice in the chart mentioned earlier, TE simply casts the computation layers into FP8, while everything else is in FP32. As a result this winds up utilizing the most memory but does so with the benefit of guaranteeing the least amount of loss in end accuracy during training. 

## `MS-AMP`

MS-AMP takes a different approach to `TransformersEngine` by providing three different optimization levels to convert more operations in FP8 or FP16.

* The base optimization level (`O1`), passes communications of the weights (such as in DDP) in FP8, stores the weights of the model in FP16, and leaves the optimizer states in FP32. The main benefit of this optimization level is that we can reduce the communication bandwidth by essentially half. Additionally, more GPU memory is saved due to 1/2 of everything being cast in FP8, and the weights being cast to FP16. Notably, both the optimizer states remain in FP32.

* The second optimization level (`O2`) improves upon this by also reducing the precision of the optimizer states. One is in FP8 while the other is in FP16. Generally it's been shown that this will only provide a net-gain of no degraded end accuracy, increased training speed, and reduced memory as now every state is either in FP16 or FP8. 

* Finally, MS-AMP has a third optimization level (`O3`) which helps during DDP scenarios such as DeepSpeed. The weights of the model in memory are fully cast to FP8, and the master weights are now stored in FP16. This fully reduces memory by the highest factor as now not only is almost everything in FP8, only two states are left in FP16. Currently, only DeepSpeed versions up through 0.9.2 are supported, so this capability is not included in the 🤗 Accelerate integration

## Combining the two

More experiments need to be performed but it's been noted that combining both MS-AMP and TransformersEngine can lead to the highest throughput by relying on NVIDIA's optimized FP8 operators and utilizing how MS-AMP reduces the memory overhead.