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

🤗 Accelerate provides integrations to train on lower precision methods using specified supported hardware through the `TransformersEngine` and `MS-AMP` packages. This documentation will help guide you through what hardware is supported, how to configure your [`Accelerator`] to leverage the low precision methods, and what you can expect when training. 

## What training on FP8 means

To explore more of the nitty-gritty in training in FP8 with PyTorch and 🤗 Accelerate, check out the [concept_guide](../concept_guides/low_precision_training.md) on why this can be difficult. But essentially rather than training in BF16, some (or all) aspects of training a model can be performed using 8 bits instead of 16. The challenge is doing so without degrading final performance. 

This is only enabled on specific NVIDIA hardware, namely:

* Anything after the 3000 series consumer graphics cards (such as the 4090)
* Hopper-based GPU architectures (such as the `H100` and `H200`)

What this will result in is some gain in the memory used (as we've cut the needed memory in half for some parts of training) and an increase in throughput *should* be seen as well for larger models that can replace certain layers with FP8-enabled ones.

## Configuring the Accelerator

Currently two different backends for FP8 are supported (`TransformersEngine` and `MS-AMP`), each with different capabilities and configurations. 

To use either, the same core API is used. Just pass `mixed_precision="fp8"` to either the [`Accelerator`], during `accelerate config` when prompted about mixed precision, or as part of your `config.yaml` file in the `mixed_precision` key:

```{python}
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp8")
```

By default, if `MS-AMP` is available in your environment, 🤗 Accelerate will automatically utilize it as a backend. To specify it yourself (and customize other parts of the FP8 mixed precision setup), you can utilize the [`utils.FP8RecipeKwargs`]:

```{python}
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="msamp")]
# Or to specify the backend as `TransformersEngine` even if MS-AMP is installed
# kwargs = [FP8RecipeKwargs(backend="te")]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## Configuring MS-AMP

Of the two, `MS-AMP` is traditionally the easier one to configure as there is only a single argument: the optimization level. 

Currently two levels of optimization are supported in the 🤗 Accelerate integration, `"O1"` and `"O2"` (using the letter 'o', not zero). 

* `"O1"` will cast the weight gradients and `all_reduce` communications to happen in 8-bit, while the rest are done in 16 bit. This reduces the general GPU memory usage and speeds up communication bandwidths.
* `"O2"` will also cast first-order optimizer states into 8 bit, while the second order states are in FP16. (Currently just the `Adam` optimizer is supported). This tries it's best to minimize final accuracy degradation and will save the highest potential memory.

To specify an optimization level, pass it to the `FP8KwargsHandler` by setting the `optimization_level` argument:

```{python}
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="msamp", optimization_level="O2")]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## Configuring TransformersEngine

TransformersEngine has much more available for customizing how and what FP8 calculations are performed. A full list of supported arguments and what they mean are available in [NVIDIA's documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html), however they are restated as part of [`FP8KwargsHandler`]'s docstring for your convience. 

🤗 Accelerate tries to set sensible defaults, but exploring and tweaking the various parameters yourself can lead to better performance potentially.

To use it, specify `backend="te"` and modify any of the arguments you want as part of your kwarg handler:

```{python}
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="te", ...)]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## Futher Reading

To learn more about training in FP8 please check out the following resources:

* [Our concept guide](../concept_guides/low_precision_training.md) detailing into more about both TransformersEngine and MS-AMP
* [The `transformers-engine` documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html)
* [The `MS-AMP` documentation](https://azure.github.io/MS-AMP/docs/)