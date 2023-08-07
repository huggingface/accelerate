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

# Performing gradient accumulation with 🤗 Accelerate

Gradient accumulation is a technique where you can train on bigger batch sizes than 
your machine would normally be able to fit into memory. This is done by accumulating gradients over
several batches, and only stepping the optimizer after a certain number of batches have been performed.

While technically standard gradient accumulation code would work fine in a distributed setup, it is not the most efficient
method for doing so and you may experience considerable slowdowns!

In this tutorial you will see how to quickly setup gradient accumulation and perform it with the utilities provided in 🤗 Accelerate,
which can total to adding just one new line of code!

This example will use a very simplistic PyTorch training loop that performs gradient accumulation every two batches:

```python
device = "cuda"
model.to(device)

gradient_accumulation_steps = 2

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (index + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## Converting it to 🤗 Accelerate

First the code shown earlier will be converted to utilize 🤗 Accelerate without the special gradient accumulation helper:

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for index, batch in enumerate(training_dataloader):
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      loss = loss / gradient_accumulation_steps
+     accelerator.backward(loss)
      if (index+1) % gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```

<Tip warning={true}>

  In its current state, this code is not going to perform gradient accumulation efficiently due to a process called gradient synchronization. Read more about that in the [Concepts tutorial](../concept_guides/gradient_synchronization)!

</Tip>

## Letting 🤗 Accelerate handle gradient accumulation

All that is left now is to let 🤗 Accelerate handle the gradient accumulation for us. To do so you should pass in a `gradient_accumulation_steps` parameter to [`Accelerator`], dictating the number 
of steps to perform before each call to `step()` and how to automatically adjust the loss during the call to [`~Accelerator.backward`]:

```diff
  from accelerate import Accelerator
- accelerator = Accelerator()
+ accelerator = Accelerator(gradient_accumulation_steps=2)
```

Alternatively, you can pass in a `gradient_accumulation_plugin` parameter to the [`Accelerator`] object's `__init__`, which will allow you to further customize the gradient accumulation behavior. 
Read more about that in the [GradientAccumulationPlugin](../package_reference/accelerator#accelerate.utils.GradientAccumulationPlugin) docs.

From here you can use the [`~Accelerator.accumulate`] context manager from inside your training loop to automatically perform the gradient accumulation for you!
You just wrap it around the entire training part of our code: 

```diff
- for index, batch in enumerate(training_dataloader):
+ for batch in training_dataloader:
+     with accelerator.accumulate(model):
          inputs, targets = batch
          outputs = model(inputs)
```

You can remove all the special checks for the step number and the loss adjustment:

```diff
- loss = loss / gradient_accumulation_steps
  accelerator.backward(loss)
- if (index+1) % gradient_accumulation_steps == 0:
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad()
```

As you can see the [`Accelerator`] is able to keep track of the batch number you are on and it will automatically know whether to step through the prepared optimizer and how to adjust the loss. 

<Tip>
Typically with gradient accumulation, you would need to adjust the number of steps to reflect the change in total batches you are 
training on. 🤗 Accelerate automagically does this for you by default. Behind the scenes we instantiate a GradientAccumulationPlugin configured to do this.
</Tip>

## The finished code

Below is the finished implementation for performing gradient accumulation with 🤗 Accelerate

```python
from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
for batch in training_dataloader:
    with accelerator.accumulate(model):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

<Tip warning={true}>

It's important that **only one forward/backward** should be done inside the context manager `with accelerator.accumulate(model)`.

</Tip>


To learn more about what magic this wraps around, read the [Gradient Synchronization concept guide](../concept_guides/gradient_synchronization)


## Self-contained example

Here is a self-contained example that you can run to see gradient accumulation in action with 🤗 Accelerate:

```python
import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader

# seed
set_seed(0)

# define toy inputs and labels
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = torch.tensor([2., 4., 6., 8., 10., 12., 14., 16.])
gradient_accumulation_steps = 4
batch_size = len(x) // gradient_accumulation_steps

# define dataset and dataloader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size)

# define model, optimizer and loss function
model = torch.zeros((1, 1), requires_grad=True)
model_clone = copy.deepcopy(model)
criterion = torch.nn.MSELoss()
model_optimizer = torch.optim.SGD([model], lr=0.02)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, model_optimizer, dataloader = accelerator.prepare(model, model_optimizer, dataloader)
model_clone_optimizer = torch.optim.SGD([model_clone], lr=0.02)
print(f"initial model weight is {model.mean().item():.5f}")
print(f"initial model weight is {model_clone.mean().item():.5f}")
for i, (inputs, labels) in enumerate(dataloader):
    with accelerator.accumulate(model):
        inputs = inputs.view(-1, 1)
        print(i, inputs.flatten())
        labels = labels.view(-1, 1)
        outputs = inputs @ model
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        model_optimizer.step()
        model_optimizer.zero_grad()
loss = criterion(x.view(-1, 1) @ model_clone, y.view(-1, 1))
model_clone_optimizer.zero_grad()
loss.backward()
model_clone_optimizer.step()
print(f"w/ accumulation, the final model weight is {model.mean().item():.5f}")
print(f"w/o accumulation, the final model weight is {model_clone.mean().item():.5f}")
```
```
initial model weight is 0.00000
initial model weight is 0.00000
0 tensor([1., 2.])
1 tensor([3., 4.])
2 tensor([5., 6.])
3 tensor([7., 8.])
w/ accumulation, the final model weight is 2.04000
w/o accumulation, the final model weight is 2.04000
```
