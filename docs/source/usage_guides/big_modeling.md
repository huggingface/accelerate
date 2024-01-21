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

# Handling big models for inference

One of the biggest advancements 🤗 Accelerate provides is the concept of [large model inference](../concept_guides/big_model_inference) wherein you can perform *inference* on models that cannot fully fit on your graphics card. 

This tutorial will be broken down into two parts showcasing how to use both 🤗 Accelerate and 🤗 Transformers (a higher API-level) to make use of this idea.

## Using 🤗 Accelerate

For these tutorials, we'll assume a typical workflow for loading your model in such that:

```py
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

Note that here we assume that `ModelClass` is a model that takes up more video-card memory than what can fit on your device (be it `mps` or `cuda`).

The first step is to init an empty skeleton of the model which won't take up any RAM using the [`init_empty_weights`] context manager:

```py
from accelerate import init_empty_weights
with init_empty_weights():
    my_model = ModelClass(...)
```

With this `my_model` currently is "parameterless", hence leaving the smaller footprint than what one would normally get loading this onto the CPU directly. 

Next we need to load in the weights to our model so we can perform inference.

For this we will use [`load_checkpoint_and_dispatch`], which as the name implies will load a checkpoint inside your empty model and dispatch the weights for each layer across all the devices you have available (GPU/MPS and CPU RAM). 

To determine how this `dispatch` can be performed, generally specifying `device_map="auto"` will be good enough as 🤗 Accelerate
will attempt to fill all the space in your GPU(s), then loading them to the CPU, and finally if there is not enough RAM it will be loaded to the disk (the absolute slowest option). 

<Tip>

For more details on designing your own device map, see this section of the [concept guide](../concept_guide/big_model_inference#designing-a-device-map)

</Tip>

See an example below:

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)
```

<Tip>

    If there are certain "chunks" of layers that shouldn't be split, you can pass them in as `no_split_module_classes`. Read more about it [here](../concept_guides/big_model_inference#loading-weights)

</Tip>

<Tip>

    Also to save on memory (such as if the `state_dict` will not fit in RAM), a model's weights can be divided and split into multiple checkpoint files. Read more about it [here](../concept_guides/big_model_inference#sharded-checkpoints)

</Tip>

Now that the model is dispatched fully, you can perform inference as normal with the model:

```py
input = torch.randn(2,3)
input = input.to("cuda")
output = model(input)
```

What will happen now is each time the input gets passed through a layer, it will be sent from the CPU to the GPU (or disk to CPU to GPU), the output is calculated, and then the layer is pulled back off the GPU going back down the line. While this adds some overhead to the inference being performed, through this method it is possible to run **any size model** on your system, as long as the largest layer is capable of fitting on your GPU. 

<Tip>

    Multiple GPUs can be utilized, however this is considered "model parallelism" and as a result only one GPU will be active at a given moment, waiting for the prior one to send it the output. You should launch your script normally with `python`
    and not need `torchrun`, `accelerate launch`, etc.

</Tip>

For a visual representation of this, check out the animation below:

<Youtube id="MWCSGj9jEAo" />

### Complete Example

Below is the full example showcasing what we performed above:

```py
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = MyModel(...)

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)

input = torch.randn(2,3)
input = input.to("cuda")
output = model(input)
```

## Using 🤗 Transformers, 🤗 Diffusers, and other 🤗 Open Source Libraries

Libraries that support 🤗 Accelerate big model inference include all of the earlier logic in their `from_pretrained` constructors. 

These operate by specifying a string representing the model to download from the [🤗 Hub](https://hf.co/models) and then denoting `device_map="auto"` along with a few extra parameters. 

As a brief example, we will look at using `transformers` and loading in Big Science's T0pp model. 

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

After loading the model in, the initial steps from before to prepare a model have all been done and the model is fully
ready to make use of all the resources in your machine. Through these constructors, you can also save *more* memory by
specifying the precision the model is loaded into as well, through the `torch_dtype` parameter, such as:

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto", torch_dtype=torch.float16)
```

To learn more about this, check out the 🤗 Transformers documentation available [here](https://huggingface.co/docs/transformers/main/en/main_classes/model#large-model-loading).

## Where to go from here

For a much more detailed look at big model inference, be sure to check out the [Conceptual Guide on it](../concept_guides/big_model_inference)
