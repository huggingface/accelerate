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

# TPU training

A [TPU (Tensor Processing Unit)](https://cloud.google.com/tpu/docs/intro-to-tpu) is a type of hardware specifically designed for training models efficiently. Accelerate supports TPU training, but there are a few things you should be aware of, namely graph compilation. This tutorial briefly discusses compilation, and for more details, take a look at the [Training on TPUs with Accelerate](../concept_guides/training_tpu) guide.

## Compilation

A TPU creates a graph of all the operations in the training step such as the forward pass, backward pass and optimizer step. This is why the first training step always takes a while because building and compiling this graph takes time. But once compilation is complete, it is cached and all subsequent steps are much faster.

The key is to avoid compiling your code again or else training is super slow. This means all your operations must be exactly the same:

* all tensors in your batches must have the same length (for example, no dynamic padding for NLP tasks)
* your code must be static (for example, no layers with for loops that have different lengths depending on the input such as a LSTM)

## Weight tying

A common language model design is to tie the weights of the embedding and softmax layers. However, moving the model to a TPU (either yourself or passing it to the [`~Accelerator.prepare`] method) breaks the weight tying and you'll need to retie the weights.

To add special behavior (like weight tying) in your script for TPUs, set [`~Accelerator.distributed_type`] to `DistributedType.TPU` first. Then you can use the [`~transformers.PreTrainedModel.tie_weights`] method to tie the weights.

```py
if accelerator.distributed_type == DistributedType.TPU:
    model.tie_weights()
```
