<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Comparing performance between different device setups

Evaluating and comparing the performance from different setups can be quite tricky if you don't know what to look for.
For example, you cannot run the same script with the same batch size across TPU, multi-GPU, and single-GPU with Accelerate 
and expect your results to line up. 

But why?

There are three reasons for this that this tutorial will cover: 

1. **Setting the right seeds**
2. **Observed Batch Sizes**
3. **Learning Rates**

## Setting the Seed 

While this issue has not come up as much, make sure to use [`utils.set_seed`] to fully set the seed in all distributed cases so training will be reproducible:

```python
from accelerate.utils import set_seed

set_seed(42)
```

Why is this important? Under the hood this will set **5** different seed settings:

```python
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if is_tpu_available():
        xm.set_rng_state(seed)
```

The random state, numpy's state, torch, torch's cuda state, and if TPUs are available torch_xla's cuda state.

## Observed Batch Sizes 

When training with Accelerate, the batch size passed to the dataloader is the **batch size per GPU**. What this entails is 
a batch size of 64 on two GPUs is truly a batch size of 128. As a result, when testing on a single GPU this needs to be accounted for,
as well as similarly for TPUs. 

The below table can be used as a quick reference to try out different batch sizes:

<Tip>

In this example, there are two GPUs for "Multi-GPU" and a TPU pod with 8 workers

</Tip>

| Single GPU Batch Size | Multi-GPU Equivalent Batch Size | TPU Equivalent Batch Size |
|-----------------------|---------------------------------|---------------------------|
| 256                   | 128                             | 32                        |
| 128                   | 64                              | 16                        |
| 64                    | 32                              | 8                         |
| 32                    | 16                              | 4                         |

## Learning Rates 

As noted in multiple sources[[1](https://aws.amazon.com/blogs/machine-learning/scalable-multi-node-deep-learning-training-using-gpus-in-the-aws-cloud/)][[2](https://docs.nvidia.com/clara/tlt-mi_archive/clara-train-sdk-v2.0/nvmidl/appendix/training_with_multiple_gpus.html)], the learning rate should be scaled *linearly* based on the number of devices present. The below 
snippet shows doing so with Accelerate:

<Tip>

Since users can have their own learning rate schedulers defined, we leave this up to the user to decide if they wish to scale their 
learning rate or not.
 
</Tip>

```python
learning_rate = 1e-3
accelerator = Accelerator()
learning_rate *= accelerator.num_processes

optimizer = AdamW(params=model.parameters(), lr=learning_rate)
```

You will also find that `accelerate` will step the learning rate based on the number of processes being trained on. This is because 
of the observed batch size noted earlier. So in the case of 2 GPUs, the learning rate will be stepped twice as often as a single GPU
to account for the batch size being twice as large (if no changes to the batch size on the single GPU instance are made).

## Gradient Accumulation and Mixed Precision

When using gradient accumulation and mixed precision, due to how gradient averaging works (accumulation) and the precision loss (mixed precision), 
some degradation in performance is expected. This will be explicitly seen when comparing the batch-wise loss between different compute 
setups. However, the overall loss, metric, and general performance at the end of training should be _roughly_ the same.
