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
    <img src="https://raw.githubusercontent.com/huggingface/accelerate/main/docs/source/imgs/accelerate_logo.png" width="400"/>
    <br>
<p>

<p align="center">
    <!-- Uncomment when CircleCI is set up
    <a href="https://circleci.com/gh/huggingface/accelerate">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    -->
    <a href="https://github.com/huggingface/accelerate/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/accelerate.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/accelerate/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/accelerate/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/accelerate/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/accelerate.svg">
    </a>
    <a href="https://github.com/huggingface/accelerate/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

<h3 align="center">
<p>Run your *raw* PyTorch training script on any kind of device
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://raw.githubusercontent.com/huggingface/accelerate/main/docs/source/imgs/course_banner.png"></a>
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
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```

As you can see in this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting (single CPU, single GPU, multi-GPUs and TPUs) as well as with or without mixed precision (fp8, fp16, bf16).

In particular, the same code can then be run without modification on your local machine for debugging or your training environment.

🤗 Accelerate even handles the device placement for you (which requires a few more changes to your code, but is safer in general), so you can even simplify your training loop further:

```diff
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

- device = 'cpu'
+ accelerator = Accelerator()

- model = torch.nn.Transformer().to(device)
+ model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
-         source = source.to(device)
-         targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
```

Want to learn more? Check out the [documentation](https://huggingface.co/docs/accelerate) or have a look at our [examples](https://github.com/huggingface/accelerate/tree/main/examples).

## Launching script

🤗 Accelerate also provides an optional CLI tool that allows you to quickly configure and test your training environment before launching the scripts. No need to remember how to use `torch.distributed.run` or to write a specific launcher for TPU training!
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

This CLI tool is **optional**, and you can still use `python my_script.py` or `python -m torchrun my_script.py` at your convenience.

You can also directly pass in the arguments you would to `torchrun` as arguments to `accelerate launch` if you wish to not run` accelerate config`.

For example, here is how to launch on two GPUs:

```bash
accelerate launch --multi_gpu --num_processes 2 examples/nlp_example.py
```

To learn more, check the CLI documentation available [here](https://huggingface.co/docs/accelerate/package_reference/cli).

## Launching multi-CPU run using MPI

🤗 Here is another way to launch multi-CPU run using MPI. You can learn how to install Open MPI on [this page](https://www.open-mpi.org/faq/?category=building#easy-build). You can use Intel MPI or MVAPICH as well.
Once you have MPI setup on your cluster, just run:

```bash
mpirun -np 2 python examples/nlp_example.py
```

## Launching training using DeepSpeed

🤗 Accelerate supports training on single/multiple GPUs using DeepSpeed. To use it, you don't need to change anything in your training code; you can set everything using just `accelerate config`. However, if you desire to tweak your DeepSpeed related args from your Python script, we provide you the `DeepSpeedPlugin`.

```python
from accelerate import Accelerator, DeepSpeedPlugin

# deepspeed needs to know your gradient accumulation steps beforehand, so don't forget to pass it
# Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)

# How to save your 🤗 Transformer?
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
```

Note: DeepSpeed support is experimental for now. In case you get into some problem, please open an issue.

## Launching your training from a notebook

🤗 Accelerate also provides a `notebook_launcher` function you can use in a notebook to launch a distributed training. This is especially useful for Colab or Kaggle notebooks with a TPU backend. Just define your training loop in a `training_function` then in your last cell, add:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

An example can be found in [this notebook](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)

## Why should I use 🤗 Accelerate?

You should use 🤗 Accelerate when you want to easily run your training scripts in a distributed environment without having to renounce full control over your training loop. This is not a high-level framework above PyTorch, just a thin wrapper so you don't have to learn a new library. In fact, the whole API of 🤗 Accelerate is in one class, the `Accelerator` object.

## Why shouldn't I use 🤗 Accelerate?

You shouldn't use 🤗 Accelerate if you don't want to write a training loop yourself. There are plenty of high-level libraries above PyTorch that will offer you that, 🤗 Accelerate is not one of them.

## Frameworks using 🤗 Accelerate

If you like the simplicity of 🤗 Accelerate but would prefer a higher-level abstraction around its capabilities, some frameworks and libraries that are built on top of 🤗 Accelerate are listed below:

* [Animus](https://github.com/Scitator/animus) is a minimalistic framework to run machine learning experiments. Animus highlights common "breakpoints" in ML experiments and provides a unified interface for them within [IExperiment](https://github.com/Scitator/animus/blob/main/animus/core.py#L76).
* [Catalyst](https://github.com/catalyst-team/catalyst#getting-started) is a PyTorch framework for Deep Learning Research and Development. It focuses on reproducibility, rapid experimentation, and codebase reuse so you can create something new rather than write yet another train loop. Catalyst provides a [Runner](https://catalyst-team.github.io/catalyst/api/core.html#runner) to connect all parts of the experiment: hardware backend, data transformations, model training, and inference logic.
* [fastai](https://github.com/fastai/fastai#installing) is a PyTorch framework for Deep Learning that simplifies training fast and accurate neural nets using modern best practices. fastai provides a [Learner](https://docs.fast.ai/learner.html#Learner) to handle the training, fine-tuning, and inference of deep learning algorithms.
* [Finetuner](https://github.com/jina-ai/finetuner) is a service that enables models to create higher-quality embeddings for semantic search, visual similarity search, cross-modal text<->image search, recommendation systems, clustering, duplication detection, anomaly detection, or other uses.
* [InvokeAI](https://github.com/invoke-ai/InvokeAI) is a creative engine for Stable Diffusion models, offering industry-leading WebUI, terminal usage support, and serves as the foundation for many commercial products.
* [Kornia](https://kornia.readthedocs.io/en/latest/get-started/introduction.html) is a differentiable library that allows classical computer vision to be integrated into deep learning models. Kornia provides a [Trainer](https://kornia.readthedocs.io/en/latest/x.html#kornia.x.Trainer) with the specific purpose to train and fine-tune the supported deep learning algorithms within the library.
* [Open Assistant](https://projects.laion.ai/Open-Assistant/) is a chat-based assistant that understands tasks, can interact with their party systems, and retrieve information dynamically to do so. 
* [pytorch-accelerated](https://github.com/Chris-hughes10/pytorch-accelerated) is a lightweight training library, with a streamlined feature set centered around a general-purpose [Trainer](https://pytorch-accelerated.readthedocs.io/en/latest/trainer.html), that places a huge emphasis on simplicity and transparency; enabling users to understand exactly what is going on under the hood, but without having to write and maintain the boilerplate themselves!
* [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is an open-source browser-based easy-to-use interface based on the Gradio library for Stable Diffusion.
* [torchkeras](https://github.com/lyhue1991/torchkeras) is a simple tool for training pytorch model just in a keras style, a dynamic and beautiful plot is provided in notebook to monitor your loss or metric.
* [transformers](https://github.com/huggingface/transformers) as a tool for helping train state-of-the-art machine learning models in PyTorch, Tensorflow, and JAX. (Accelerate is the backend for the PyTorch side).


## Installation

This repository is tested on Python 3.8+ and PyTorch 1.10.0+

You should install 🤗 Accelerate in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install PyTorch: refer to the [official installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform. Then 🤗 Accelerate can be installed using pip as follows:

```bash
pip install accelerate
```

## Supported integrations

- CPU only
- multi-CPU on one node (machine)
- multi-CPU on several nodes (machines)
- single GPU
- multi-GPU on one node (machine)
- multi-GPU on several nodes (machines)
- TPU
- FP16/BFloat16 mixed precision
- FP8 mixed precision with [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- DeepSpeed support (Experimental)
- PyTorch Fully Sharded Data Parallel (FSDP) support (Experimental)
- Megatron-LM support (Experimental)

## Citing 🤗 Accelerate

If you use 🤗 Accelerate in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{accelerate,
  title =        {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
  author =       {Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/accelerate}},
  year =         {2022}
}
```
