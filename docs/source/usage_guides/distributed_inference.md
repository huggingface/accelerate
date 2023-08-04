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

# Distributed Inference with 🤗 Accelerate

Distributed inference is a common use case, especially with natural language processing (NLP) models. Users often want to
send a number of different prompts, each to a different GPU, and then get the results back. This also has other cases
outside of just NLP, however for this tutorial we will focus on just this idea of each GPU receiving a different prompt,
and then returning the results.

## The Problem

Normally when doing this, users send the model to a specific device to load it from the CPU, and then move each prompt to a different device. 

A basic pipeline using the `diffusers` library might look something like so:

```python
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
```
Followed then by performing inference based on the specific prompt:

```python
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    pipe.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    result = pipe(prompt).images[0]
    result.save(f"result_{rank}.png")
```
One will notice how we have to check the rank to know what prompt to send, which can be a bit tedious.

A user might then also think that with 🤗 Accelerate, using the `Accelerator` to prepare a dataloader for such a task might also be 
a simple way to manage this. (To learn more, check out the relvent section in the [Quick Tour](../quicktour#distributed-evaluation))

Can it manage it? Yes. Does it add unneeded extra code however: also yes.

## The Solution

With 🤗 Accelerate, we can simplify this process by using the [`Accelerator.split_between_processes`] context manager (which also exists in `PartialState` and `AcceleratorState`). 
This function will automatically split whatever data you pass to it (be it a prompt, a set of tensors, a dictionary of the prior data, etc.) across all the processes (with a potential
to be padded) for you to use right away.

Let's rewrite the above example using this context manager:

```python
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipe.to(distributed_state.device)

# Assume two processes
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipe(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

And then to launch the code, we can use the 🤗 Accelerate:

If you have generated a config file to be used using `accelerate config`:

```bash
accelerate launch distributed_inference.py
```

If you have a specific config file you want to use:

```bash
accelerate launch --config_file my_config.json distributed_inference.py
```

Or if don't want to make any config files and launch on two GPUs:

> Note: You will get some warnings about values being guessed based on your system. To remove these you can do `accelerate config default` or go through `accelerate config` to create a config file.

```bash
accelerate launch --num_processes 2 distributed_inference.py
```

We've now reduced the boilerplate code needed to split this data to a few lines of code quite easily.

But what if we have an odd distribution of prompts to GPUs? For example, what if we have 3 prompts, but only 2 GPUs? 

Under the context manager, the first GPU would receive the first two prompts and the second GPU the third, ensuring that 
all prompts are split and no overhead is needed.

*However*, what if we then wanted to do something with the results of *all the GPUs*? (Say gather them all and perform some kind of post processing)
You can pass in `apply_padding=True` to ensure that the lists of prompts are padded to the same length, with extra data being taken 
from the last sample. This way all GPUs will have the same number of prompts, and you can then gather the results.

<Tip>

This is only needed when trying to perform an action such as gathering the results, where the data on each device 
needs to be the same length. Basic inference does not require this.

</Tip>

For instance:

```python
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipe.to(distributed_state.device)

# Assume two processes
with distributed_state.split_between_processes(["a dog", "a cat", "a chicken"], apply_padding=True) as prompt:
    result = pipe(prompt).images
```

On the first GPU, the prompts will be `["a dog", "a cat"]`, and on the second GPU it will be `["a chicken", "a chicken"]`.
Make sure to drop the final sample, as it will be a duplicate of the previous one.
