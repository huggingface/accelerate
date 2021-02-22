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
    <!-- Uncomment when repo is public
    <a href="https://github.com/huggingface/accelerate/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/accelerate.svg?color=blue">
    </a>
    -->
    <!-- Uncomment when doc is online
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    -->
    <!-- Uncomment when repo is public
    <a href="https://github.com/huggingface/accelerate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/accelerate.svg">
    </a>
    -->
    <a href="https://github.com/huggingface/accelerate/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p>Run your *raw* PyTorch training script on any kind of device
</h3>

🤗 Accelerate was created for PyTorch users who like to write the training loop of PyTorch models but are reluctant to write and maintain the boiler code needed to use multi-GPUs/TPU/fp16.

🤗 Accelerate abstracts exactly and only the boiler code related to multi-GPUs/TPU/fp16 and let the rest of your code unchanged.

Here is an example:

<table>
<tr>
<th> Original training code <br> (CPU or mono-GPU only)</th>
<th> With Accelerate <br> (CPU/GPU/multi-GPUs/TPUs/fp16) </th>
</tr>
<tr>
<td>

```python
import torch
import torch.nn.functional as F
from datasets import load_dataset



device = 'cpu'

model = torch.nn.Transformer().to(device)
optim = torch.optim.Adam(
    model.parameters()
)

dataset = load_dataset('my_dataset')
data = torch.utils.data.Dataloader(
    dataset
)




model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source, targets)
        loss = F.cross_entropy(
            output, targets
        )

        loss.backward()

        optimizer.step()
```

</td>
<td>

```python
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset

+ from accelerate import Accelerator
+ accelerator = Accelerator(device_placement=False)
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optim = torch.optim.Adam(
      model.parameters()
  )

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.Dataloader(
      dataset
  )

+ model, optim, data = accelerator.prepare(
      model, optim, data
  )

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source, targets)
          loss = F.cross_entropy(
              output, targets
          )

+         accelerate.backward(loss)

          optimizer.step()
```

</td>
</tr>
</table>

As you can see on this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting (single CPU, single GPU, multi-GPUs and TPUs) as well as with or without mixed precision (fp16).

The same code can then in paticular run without modification on your local machine for debugging or your training environment.

🤗 Accelerate also provides a CLI tool that allows you to quickly configure and test your training environment then launch the scripts.

## Easy to integrate

A traditional training loop in PyTorch looks like this:

```python
my_model.to(device)

for batch in my_training_dataloader:
    my_optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = my_model(inputs)
    loss = my_loss_function(outputs, targets)
    loss.backward()
    my_optimizer.step()
```

Changing it to work with accelerate is really easy and only adds a few lines of code:

```python
from accelerate import Accelerator

accelerator = Accelerator(device_placement=False)
# Use the device given by the `accelerator` object.
device = accelerator.device
my_model.to(device)
# Pass every important object (model, optimizer, dataloader) to `accelerator.prepare`
my_model, my_optimizer, my_training_dataloader = accelerate.prepare(
    my_model, my_optimizer, my_training_dataloader
)

for batch in my_training_dataloader:
    my_optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = my_model(inputs)
    loss = my_loss_function(outputs, targets)
    # Just a small change for the backward instruction
    accelerate.backward(loss)
    my_optimizer.step()
```

and with this, your script can now run in a distributed environment (multi-GPU, TPU).

You can even simplify your script a bit by letting 🤗 Accelerate handle the device placement for you (which is safer, especially for TPU training):

```python
from accelerate import Accelerator

accelerator = Accelerator()
# Pass every important object (model, optimizer, dataloader) to `accelerator.prepare`
my_model, my_optimizer, my_training_dataloader = accelerate.prepare(
    my_model, my_optimizer, my_training_dataloader
)

for batch in my_training_dataloader:
    my_optimizer.zero_grad()
    inputs, targets = batch
    outputs = my_model(inputs)
    loss = my_loss_function(outputs, targets)
    # Just a small change for the backward instruction
    accelerate.backward(loss)
    my_optimizer.step()
```

Checkout our [example](/examples) to see how it can be used on a variety of scripts!

## Script launcher

No need to remember how to use `torch.distributed.launch` or to write a specific launcher for TPU training! 🤗 Accelerate comes with a CLI tool that will make your life easier when launching distributed scripts.

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
accelerate launch examples/glue_example.py --task_name mrpc --model_name_or_path bert-base-cased
```

## Why should I use 🤗 Accelerate?

You should use 🤗 Accelerate when you want to easily run your training scripts in a distributed environment without having to renounce full control over your training loop. This is not a high-level framework above PyTorch, just a thin wrapper so you don't have to learn a new library, In fact the whole API of 🤗 Accelerate is in one class, the `Accelerator` object.

## Why shouldn't use 🤗 Accelerate?

You shouldn't use 🤗 Accelerate if you don't want to write a training loop yourself. There are plenty of high-level libraries above PyTorch that will offer you that, 🤗 Accelerate is not one of them.

## Installation

This repository is tested on Python 3.6+ and PyTorch 1.4.0+

You should install 🤗 Accelerate in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install PyTorch: refer to the [official installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform. Then 🤗 Accelerate can be installed using pip as follows:

```bash
pip install accelerate
```
