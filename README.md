<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <br>
    <img src="docs/source/imgs/accelerate_logo.png" width="400"/>
    <br>
<p>

<p align="center">
    <!-- Uncomment when CircleCI is setup
    <a href="https://circleci.com/gh/huggingface/accelerate">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    -->
    <a href="https://github.com/huggingface/accelerate/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/accelerate.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/accelerate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/accelerate.svg">
    </a>
    <a href="https://github.com/huggingface/accelerate/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p>Run your *raw* PyTorch training script on any kind of device
</h3>

## Easy to integrate

🤗 Accelerate was created for PyTorch users who like to write the training loop of PyTorch models but are reluctant to write and maintain the boilerplate code needed to use multi-GPUs/TPU/fp16.

🤗 Accelerate abstracts exactly and only the boilerplate code related to multi-GPUs/TPU/fp16 and leaves the rest of your code unchanged.

Here is an example:

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cpu'
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optim = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optim, data = accelerator.prepare(model, optim, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

+         accelerator.backward(loss)
-         loss.backward()

          optimizer.step()
```

As you can see in this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting (single CPU, single GPU, multi-GPUs and TPUs) as well as with or without mixed precision (fp16).

In particular, the same code can then be run without modification on your local machine for debugging or your training environment.

🤗 Accelerate even handles the device placement for you (which requires a few more changes to your code, but is safer in general), so you can even simplify your training loop further:

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cpu'

+ model = torch.nn.Transformer()
- model = torch.nn.Transformer().to(device)
  optim = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optim, data = accelerator.prepare(model, optim, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
-         source = source.to(device)
-         targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

+         accelerator.backward(loss)
-         loss.backward()

          optimizer.step()
```

## Launching script

🤗 Accelerate also provides an optional CLI tool that allows you to quickly configure and test your training environment before launching the scripts. No need to remember how to use `torch.distributed.launch` or to write a specific launcher for TPU training!
On your machine(s) just run:

```bash
accelerate config
```

and answer the questions asked. This will generate a config file that will be used automatically to properly set the default options when doing

```bash
accelerate launch my_script.py --args_to_my_script
``` 

For instance, here is how you would run the GLUE example on the MRPC task (from the root of the repo):

```bash
accelerate launch examples/nlp_example.py
```

## Why should I use 🤗 Accelerate?

You should use 🤗 Accelerate when you want to easily run your training scripts in a distributed environment without having to renounce full control over your training loop. This is not a high-level framework above PyTorch, just a thin wrapper so you don't have to learn a new library, In fact the whole API of 🤗 Accelerate is in one class, the `Accelerator` object.

## Why shouldn't I use 🤗 Accelerate?

You shouldn't use 🤗 Accelerate if you don't want to write a training loop yourself. There are plenty of high-level libraries above PyTorch that will offer you that, 🤗 Accelerate is not one of them.

## Installation

This repository is tested on Python 3.6+ and PyTorch 1.4.0+

You should install 🤗 Accelerate in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install PyTorch: refer to the [official installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform. Then 🤗 Accelerate can be installed using pip as follows:

```bash
pip install accelerate
```

## Supported integrations

- CPU only
- single GPU
- multi-GPU on one node (machine)
- multi-GPU on several nodes (machines)
- TPU
- FP16 with native AMP (apex on the roadmap)
