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

Distributed inference can fall into three brackets:

1. Loading an entire model onto each GPU and sending chunks of a batch through each GPU's model copy at a time
2. Loading parts of a model onto each GPU and processing a single input at one time
3. Loading parts of a model onto each GPU and using what is called scheduled Pipeline Parallelism to combine the two prior techniques. 

We're going to go through the first and the last bracket, showcasing how to do each as they are more realistic scenarios.


## Sending chunks of a batch automatically to each loaded model

This is the most memory-intensive solution, as it requires each GPU to keep a full copy of the model in memory at a given time. 

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
a simple way to manage this. (To learn more, check out the relevant section in the [Quick Tour](../quicktour#distributed-evaluation))

Can it manage it? Yes. Does it add unneeded extra code however: also yes.


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

## Memory-efficient pipeline parallelism (experimental)

This next part will discuss using *pipeline parallelism*. This is an **experimental** API utilizing the [PiPPy library by PyTorch](https://github.com/pytorch/PiPPy/) as a native solution. 

The general idea with pipeline parallelism is: say you have 4 GPUs and a model big enough it can be *split* on four GPUs using `device_map="auto"`. With this method you can send in 4 inputs at a time (for example here, any amount works) and each model chunk will work on an input, then receive the next input once the prior chunk finished, making it *much* more efficient **and faster** than the method described earlier. Here's a visual taken from the PyTorch repository:

![PiPPy example](https://camo.githubusercontent.com/681d7f415d6142face9dd1b837bdb2e340e5e01a58c3a4b119dea6c0d99e2ce0/68747470733a2f2f692e696d6775722e636f6d2f657955633934372e706e67)

To illustrate how you can use this with Accelerate, we have created an [example zoo](https://github.com/huggingface/accelerate/tree/main/examples/inference) showcasing a number of different models and situations. In this tutorial, we'll show this method for GPT2 across two GPUs.

Before you proceed, please make sure you have the latest pippy installed by running the following:

```bash
pip install torchpippy
```

We require at least version 0.2.0. To confirm that you have the correct version, run `pip show torchpippy`.

Start by creating the model on the CPU:

```{python}
from transformers import GPT2ForSequenceClassification, GPT2Config

config = GPT2Config()
model = GPT2ForSequenceClassification(config)
model.eval()
```

Next you'll need to create some example inputs to use. These help PiPPy trace the model.

<Tip warning={true}>
    However you make this example will determine the relative batch size that will be used/passed
    through the model at a given time, so make sure to remember how many items there are!
</Tip>

```{python}
input = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 1024),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)
```
Next we need to actually perform the tracing and get the model ready. To do so, use the [`inference.prepare_pippy`] function and it will fully wrap the model for pipeline parallelism automatically:

```{python}
from accelerate.inference import prepare_pippy
example_inputs = {"input_ids": input}
model = prepare_pippy(model, example_args=(input,))
```

<Tip>

    There are a variety of parameters you can pass through to `prepare_pippy`:
    
    * `split_points` lets you determine what layers to split the model at. By default we use wherever `device_map="auto" declares, such as `fc` or `conv1`.

    * `num_chunks` determines how the batch will be split and sent to the model itself (so `num_chunks=1` with four split points/four GPUs will have a naive MP where a single input gets passed between the four layer split points)

</Tip>

From here, all that's left is to actually perform the distributed inference!

<Tip warning={true}>

When passing inputs, we highly recommend to pass them in as a tuple of arguments. Using `kwargs` is supported, however, this approach is experimental.
</Tip>

```{python}
args = some_more_arguments
with torch.no_grad():
    output = model(*args)
```

When finished all the data will be on the last process only:

```{python}
from accelerate import PartialState
if PartialState().is_last_process:
    print(output)
```

<Tip>

    If you pass in `gather_output=True` to [`inference.prepare_pippy`], the output will be sent
    across to all the GPUs afterwards without needing the `is_last_process` check. This is 
    `False` by default as it incurs a communication call.
    
</Tip>

And that's it! To explore more, please check out the inference examples in the [Accelerate repo](https://github.com/huggingface/accelerate/tree/main/examples/inference) and our [documentation](../package_reference/inference) as we work to improving this integration. 
