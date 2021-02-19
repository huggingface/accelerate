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
<th> Original training code (CPU or mono-GPU only)</th>
<th> With Accelerate for CPU/GPU/multi-GPUs/TPUs/fp16 </th>
</tr>
<tr>
<td>

```python
import torch
import torch.nn.functional as F
from datasets import load_dataset



device = 'cpu'

model = torch.nn.Transformer().to(device)
optim = torch.optim.Adam(model.parameters())

dataset = load_dataset('my_dataset')
data = torch.utils.data.Dataloader(dataset)




model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source, targets)
        loss = F.cross_entropy(output, targets)

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
+ accelerator = Accelerator()
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optim = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.Dataloader(dataset)

+ model, optim, data = accelerator.prepare(
                            model, optim, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source, targets)
          loss = F.cross_entropy(output, targets)

+         accelerate.backward(loss)

          optimizer.step()
```

</td>
</tr>
</table>

As you can see on this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting (single CPU, single GPU, multi-GPUs and TPUs) as well as with or without mixed precision (fp16).

The same code can then in paticular run without modification on your local machine for debugging or your training environment.

🤗 Accelerate also provides a CLI tool that allows you to quickly configure and test your training environment then launch the scripts.



## Installation

Install PyTorch, then

```bash
git clone https://github.com/huggingface/accelerate.git
cd accelerate
pip install -e .
```

## Tests

### Using the accelerate CLI

Create a default config for your environment with
```bash
accelerate config
```
then launch the GLUE example with
```bash
accelerate launch examples/glue_example.py --task_name mrpc --model_name_or_path bert-base-cased
```

### Traditional launchers

To run the example script on multi-GPU:
```bash
python -m torch.distributed.launch --nproc_per_node 2 --use_env examples/glue_example.py \
    --task_name mrpc --model_name_or_path bert-base-cased
```

To run the example script on TPUs:
```bash
python tests/xla_spawn.py --num_cores 8 examples/glue_example.py\
    --task_name mrpc --model_name_or_path bert-base-cased
```