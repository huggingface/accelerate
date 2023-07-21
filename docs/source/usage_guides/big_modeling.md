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

When loading a pre-trained model in PyTorch, the usual workflow looks like this:

```py
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

In plain English, those steps are:
1. Create the model with randomly initialized weights
2. Load the model weights (in a dictionary usually called a state dict) from the disk
3. Load those weights inside the model

While this works very well for regularly sized models, this workflow has some clear limitations when we deal with a huge model: in step 1, we load a full version of the model in RAM, and spend some time randomly initializing the weights (which will be discarded in step 3). In step 2, we load another full version of the model in RAM, with the pre-trained weights. If you're loading a model with 6 billion parameters, this means you will need 24GB of RAM for each copy of the model, so 48GB in total (half of it to load the model in FP16).

<Tip warning={true}>

    This API is quite new and still in its experimental stage. While we strive to provide a stable API, it's possible some small parts of the public API will change in the future.

</Tip>

## How the Process Works: A Quick Overview

<Youtube id="MWCSGj9jEAo" />

## How the Process Works: Working with Code

### Instantiating an empty model

The first tool 🤗 Accelerate introduces to help with big models is a context manager [`init_empty_weights`] that helps you initialize a model without using any RAM so that step 1 can be done on models of any size. Here is how it works:

```py
from accelerate import init_empty_weights

with init_empty_weights():
    my_model = ModelClass(...)
```

For instance:

```py
with init_empty_weights():
    model = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
```

initializes an empty model with a bit more than 100B parameters. Behind the scenes, this relies on the meta device introduced in PyTorch 1.9. During the initialization under the context manager, each time a parameter is created, it is instantly moved to that device.

<Tip warning={true}>

    You can't move a model initialized like this on CPU or another device directly, since it doesn't have any data. It's also very likely that a forward pass with that empty model will fail, as not all operations are supported on the meta device.

</Tip>

### Sharded checkpoints

It's possible your model is so big that even a single copy won't fit in RAM. That doesn't mean it can't be loaded: if you have one or several GPUs, this is more memory available to store your model. In this case, it's better if your checkpoint is split into several smaller files that we call checkpoint shards.

🤗 Accelerate will handle sharded checkpoints as long as you follow the following format: your checkpoint should be in a folder, with several files containing the partial state dicts, and there should be an index in the JSON format that contains a dictionary mapping parameter names to the file containing their weights. You can easily shard your model with [`~Accelerator.save_model`]. For instance, we could have a folder containing:

```bash
first_state_dict.bin
index.json
second_state_dict.bin
```

with index.json being the following file:

```
{
  "linear1.weight": "first_state_dict.bin",
  "linear1.bias": "first_state_dict.bin",
  "linear2.weight": "second_state_dict.bin",
  "linear2.bias": "second_state_dict.bin"
}
```

and `first_state_dict.bin` containing the weights for `"linear1.weight"` and `"linear1.bias"`, `second_state_dict.bin` the ones for `"linear2.weight"` and `"linear2.bias"`

### Loading weights

The second tool 🤗 Accelerate introduces is a function [`load_checkpoint_and_dispatch`], that will allow you to load a checkpoint inside your empty model. This supports full checkpoints (a single file containing the whole state dict) as well as sharded checkpoints. It will also automatically dispatch those weights across the devices you have available (GPUs, CPU RAM), so if you are loading a sharded checkpoint, the maximum RAM usage will be the size of the biggest shard.

If you want to use big model inference with 🤗 Transformers models, check out this [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/model#large-model-loading).

Here is how we can use this to load the [GPT2-1.5B](https://huggingface.co/marcsun13/gpt2-xl-linear-sharded) model.

Let's download the sharded version of this model.

```bash
pip install huggingface_hub
```

```py
from huggingface_hub import snapshot_download
checkpoint = "marcsun13/gpt2-xl-linear-sharded"
weights_location = snapshot_download(repo_id=checkpoint)
```

In order to initialize the model, we will use the library minGPT. 

```bash
git clone https://github.com/karpathy/minGPT.git
pip install minGPT/
```

```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    model = GPT(model_config)
```

Then, load the checkpoint we just downloaded with:

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map="auto", no_split_module_classes=['Block']
)
```

By passing `device_map="auto"`, we tell 🤗 Accelerate to determine automatically where to put each layer of the model depending on the available resources:
- first, we use the maximum space available on the GPU(s)
- if we still need space, we store the remaining weights on the CPU
- if there is not enough RAM, we store the remaining weights on the hard drive as memory-mapped tensors

`no_split_module_classes=["Block"]` indicates that the modules that are `Block` should not be split on different devices. You should set here all blocks that include a residual connection of some kind.

You can see the `device_map` that 🤗 Accelerate picked by accessing the `hf_device_map` attribute of your model:

```py
model.hf_device_map
```

```python out
{'transformer.wte': 0,
 'transformer.wpe': 0,
 'transformer.drop': 0,
 'transformer.h.0': 0,
 'transformer.h.1': 0,
 'transformer.h.2': 0,
 'transformer.h.3': 0,
 'transformer.h.4': 0,
 'transformer.h.5': 0,
 'transformer.h.6': 0,
 'transformer.h.7': 0,
 'transformer.h.8': 0,
 'transformer.h.9': 0,
 'transformer.h.10': 0,
 'transformer.h.11': 0,
 'transformer.h.12': 0,
 'transformer.h.13': 0,
 'transformer.h.14': 0,
 'transformer.h.15': 0,
 'transformer.h.16': 0, 
 'transformer.h.17': 0, 
 'transformer.h.18': 0, 
 'transformer.h.19': 0, 
 'transformer.h.20': 0, 
 'transformer.h.21': 0, 
 'transformer.h.22': 1, 
 'transformer.h.23': 1, 
 'transformer.h.24': 1, 
 'transformer.h.25': 1, 
 'transformer.h.26': 1, 
 'transformer.h.27': 1, 
 'transformer.h.28': 1, 
 'transformer.h.29': 1, 
 'transformer.h.30': 1, 
 'transformer.h.31': 1, 
 'transformer.h.32': 1, 
 'transformer.h.33': 1, 
 'transformer.h.34': 1, 
 'transformer.h.35': 1, 
 'transformer.h.36': 1, 
 'transformer.h.37': 1, 
 'transformer.h.38': 1, 
 'transformer.h.39': 1, 
 'transformer.h.40': 1, 
 'transformer.h.41': 1, 
 'transformer.h.42': 1, 
 'transformer.h.43': 1, 
 'transformer.h.44': 1, 
 'transformer.h.45': 1, 
 'transformer.h.46': 1, 
 'transformer.h.47': 1, 
 'transformer.ln_f': 1, 
 'lm_head': 1}
 ```

You can also design your `device_map` yourself if you prefer to explicitly decide where each layer should be. In this case, the command above becomes:

```py
model = load_checkpoint_and_dispatch(model, checkpoint=weights_location, device_map=my_device_map)
```

### Run the model

Now that we have done this, our model lies across several devices, and maybe the hard drive. But it can still be used as a regular PyTorch model:

```py
from mingpt.bpe import BPETokenizer
tokenizer = BPETokenizer()
inputs = tokenizer("Hello, my name is").to(0)

outputs = model.generate(x1, max_new_tokens=10, do_sample=False)[0]
tokenizer.decode(outputs.cpu().squeeze())
```

Behind the scenes, 🤗 Accelerate added hooks to the model, so that:
- at each layer, the inputs are put on the right device (so even if your model is spread across several GPUs, it works)
- for the weights offloaded on the CPU, they are put on a GPU just before the forward pass and cleaned up just after
- for the weights offloaded on the hard drive, they are loaded in RAM then put on a GPU just before the forward pass and cleaned up just after

This way, your model can run for inference even if it doesn't fit on one of the GPUs or the CPU RAM!

<Tip warning={true}>

    This only supports the inference of your model, not training. Most of the computation happens behind `torch.no_grad()` context managers to avoid spending some GPU memory with intermediate activations.

</Tip>

### Designing a device map

You can let 🤗 Accelerate handle the device map computation by setting `device_map` to one of the supported options (`"auto"`, `"balanced"`, `"balanced_low_0"`, `"sequential"`) or create one yourself if you want more control over where each layer should go.

<Tip>

    You can derive all sizes of the model (and thus compute a `device_map`) on a model that is on the meta device.

</Tip>

All the options will produce the same result when you don't have enough GPU memory to accommodate the whole model (which is to fit everything that can on the GPU, then offload weights on the CPU or even on the disk if there is not enough RAM). 

When you have more GPU memory available than the model size, here is the difference between each option:
- `"auto"` and `"balanced"` evenly split the model on all available GPUs, making it possible for you to use a batch size greater than 1.
- `"balanced_low_0"` evenly splits the model on all GPUs except the first one, and only puts on GPU 0 what does not fit on the others. This option is great when you need to use GPU 0 for some processing of the outputs, like when using the `generate` function for Transformers models
- `"sequential"` will fit what it can on GPU 0, then move on GPU 1 and so forth (so won't use the last GPUs if it doesn't need to).

<Tip>

    The options `"auto"` and `"balanced"` produce the same results for now, but the behavior of `"auto"` might change in the future if we find a strategy that makes more sense, while `"balanced"` will stay stable.

</Tip>

First note that you can limit the memory used on each GPU by using the `max_memory` argument (available in [`infer_auto_device_map`] and in all functions using it). When setting `max_memory`, you should pass along a dictionary containing the GPU identifiers (for instance `0`, `1` etc.) and the `"cpu"` key for the maximum RAM you want to use for CPU offload. The values can either be an integer (in bytes) or a string representing a number with its unit, such as `"10GiB"` or `"10GB"`.

Here is an example where we don't want to use more than 10GiB on each of the two GPUs and no more than 30GiB of CPU RAM for the model weights:

```python
from accelerate import infer_auto_device_map

device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
```

<Tip warning={true}>

    When a first allocation happens in PyTorch, it loads CUDA kernels which take about 1-2GB of memory depending on the GPU. Therefore you always have less usable memory than the actual size of the GPU. To see how much memory is actually used do `torch.ones(1).cuda()` and look at the memory usage.

    Therefore when you create memory maps with `max_memory` make sure to adjust the available memory accordingly to avoid out-of-memory errors.

</Tip>

Additionally, if you do some additional operations with your outputs without placing them back on the CPU (for instance inside the `generate` method of Transformers) and if you placed your inputs on a GPU, that GPU will consume more memory than the others (Accelerate always place the output back to the device of the input). Therefore if you would like to optimize the maximum batch size and you have many GPUs, give the first GPU less memory. For example, with BLOOM-176B on 8x80 A100 setup, the close-to-ideal map is:

```python
max_memory = {0: "30GIB", 1: "46GIB", 2: "46GIB", 3: "46GIB", 4: "46GIB", 5: "46GIB", 6: "46GIB", 7: "46GIB"}
```
as you can see we gave the remaining 7 GPUs ~50% more memory than GPU 0.

If you opt to fully design the `device_map` yourself, it should be a dictionary with keys being module names of your model and values being a valid device identifier (for instance an integer for the GPUs) or `"cpu"` for CPU offload, `"disk"` for disk offload. The keys need to cover the whole model, you can then define your device map as you wish: for instance, if your model has two blocks (let's say `block1` and `block2`) which each contain three linear layers (let's say `linear1`, `linear2` and `linear3`), a valid device map can be:

```python
device_map = {"block1": 0, "block2": 1}
```

another one that is valid could be:

```python
device_map = {"block1": 0, "block2.linear1": 0, "block2.linear2": 1, "block2.linear3": 1}
```

On the other hand, this one is not valid as it does not cover every parameter of the model:

```python
device_map = {"block1": 0, "block2.linear1": 1, "block2.linear2": 1}
```

<Tip>

    To be the most efficient, make sure your device map puts the parameters on the GPUs in a sequential manner (e.g. don't put one of the first weights on GPU 0, then weights on GPU 1 and the last weight back to GPU 0) to avoid making many transfers of data between the GPUs.

</Tip>

## Limits and further development

We are aware of the current limitations in the API:

- While this could theoretically work on just one CPU with potential disk offload, you need at least one GPU to run this API. This will be fixed in further development.
- [`infer_auto_device_map`] (or `device_map="auto"` in [`load_checkpoint_and_dispatch`]) tries to maximize GPU and CPU RAM it sees available when you execute it. While PyTorch is very good at managing GPU RAM efficiently (and giving it back when not needed), it's not entirely true with Python and CPU RAM. Therefore, an automatically computed device map might be too intense on the CPU. Move a few modules to the disk device if you get crashes due to a lack of RAM.
- [`infer_auto_device_map`] (or `device_map="auto"` in [`load_checkpoint_and_dispatch`]) attributes devices sequentially (to avoid moving things back and forth) so if your first layer is bigger than the size of the GPU you have, it will end up with everything on the CPU/Disk.
- [`load_checkpoint_and_dispatch`] and [`load_checkpoint_in_model`] do not perform any check on the correctness of your state dict compared to your model at the moment (this will be fixed in a future version), so you may get some weird errors if trying to load a checkpoint with mismatched or missing keys.
- The model parallelism used when your model is split on several GPUs is naive and not optimized, meaning that only one GPU works at a given time and the other sits idle.
- When weights are offloaded on the CPU/hard drive, there is no pre-fetching (yet, we will work on this for future versions) which means the weights are put on the GPU when they are needed and not before.
- Hard-drive offloading might be very slow if the hardware you run on does not have fast communication between disk and CPU (like NVMes).
