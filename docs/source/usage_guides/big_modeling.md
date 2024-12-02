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

# Big Model Inference

One of the biggest advancements Accelerate provides is [Big Model Inference](../concept_guides/big_model_inference), which allows you to perform inference with models that don't fully fit on your graphics card.

This tutorial will show you how to use Big Model Inference in Accelerate and the Hugging Face ecosystem.

## Accelerate

A typical workflow for loading a PyTorch model is shown below. `ModelClass` is a model that exceeds the GPU memory of your device (mps or cuda or xpu).

```py
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

With Big Model Inference, the first step is to init an empty skeleton of the model with the `init_empty_weights` context manager. This doesn't require any memory because `my_model` is "parameterless".

```py
from accelerate import init_empty_weights
with init_empty_weights():
    my_model = ModelClass(...)
```

Next, the weights are loaded into the model for inference.

The [`load_checkpoint_and_dispatch`] method loads a checkpoint inside your empty model and dispatches the weights for each layer across all available devices, starting with the fastest devices (GPU, MPS, XPU, NPU, MLU, MUSA) first before moving to the slower ones (CPU and hard drive).

Setting `device_map="auto"` automatically fills all available space on the GPU(s) first, then the CPU, and finally, the hard drive (the absolute slowest option) if there is still not enough memory.

> [!TIP]
> Refer to the [Designing a device map](../concept_guides/big_model_inference#designing-a-device-map) guide for more details on how to design your own device map.

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)
```

If there are certain “chunks” of layers that shouldn’t be split, pass them to `no_split_module_classes` (see [here](../concept_guides/big_model_inference#loading-weights) for more details).

A models weights can also be sharded into multiple checkpoints to save memory, such as when the `state_dict` doesn't fit in memory (see [here](../concept_guides/big_model_inference#sharded-checkpoints) for more details).

Now that the model is fully dispatched, you can perform inference.

```py
input = torch.randn(2,3)
input = input.to(model.device.type)
output = model(input)
```

Each time an input is passed through a layer, it is sent from the CPU to the GPU (or disk to CPU to GPU), the output is calculated, and the layer is removed from the GPU going back down the line. While this adds some overhead to inference, it enables you to run any size model on your system, as long as the largest layer fits on your GPU.

Multiple GPUs, or "model parallelism", can be utilized but only one GPU will be active at any given moment. This forces the GPU to wait for the previous GPU to send it the output. You should launch your script normally with Python instead of other tools like torchrun and accelerate launch.

> [!TIP]
> You may also be interested in *pipeline parallelism* which utilizes all available GPUs at once, instead of only having one GPU active at a time. This approach is less flexbile though. For more details, refer to the [Memory-efficient pipeline parallelism](./distributed_inference#memory-efficient-pipeline-parallelism-experimental) guide.

<Youtube id="MWCSGj9jEAo"/>

Take a look at a full example of Big Model Inference below.

```py
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = MyModel(...)

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)

input = torch.randn(2,3)
input = input.to(model.device.type)
output = model(input)
```

## Hugging Face ecosystem

Other libraries in the Hugging Face ecosystem, like Transformers or Diffusers, supports Big Model Inference in their [`~transformers.PreTrainedModel.from_pretrained`] constructors.

You just need to add `device_map="auto"` in [`~transformers.PreTrainedModel.from_pretrained`] to enable Big Model Inference.

For example, load Big Sciences T0pp 11 billion parameter model with Big Model Inference.

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

After loading the model, the empty init and smart dispatch steps from before are executed and the model is fully ready to make use of all the resources in your machine. Through these constructors, you can also save more memory by specifying the `torch_dtype` parameter to load a model in a lower precision.

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto", torch_dtype=torch.float16)
```

## Next steps

For a more detailed explanation of Big Model Inference, make sure to check out the [conceptual guide](../concept_guides/big_model_inference)!
