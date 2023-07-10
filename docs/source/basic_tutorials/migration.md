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

# Migrating your code to 🤗 Accelerate

This tutorial will detail how to easily convert existing PyTorch code to use 🤗 Accelerate!
You'll see that by just changing a few lines of code, 🤗 Accelerate can perform its magic and get you on 
your way toward running your code on distributed systems with ease!

## The base training loop

To begin, write out a very basic PyTorch training loop. 

<Tip>

    We are under the presumption that `training_dataloader`, `model`, `optimizer`, `scheduler`, and `loss_function` have been defined beforehand.

</Tip>

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## Add in 🤗 Accelerate

To start using 🤗 Accelerate, first import and create an [`Accelerator`] instance:
```python
from accelerate import Accelerator

accelerator = Accelerator()
```
[`Accelerator`] is the main force behind utilizing all the possible options for distributed training!

### Setting the right device

The [`Accelerator`] class knows the right device to move any PyTorch object to at any time, so you should
change the definition of `device` to come from [`Accelerator`]:

```diff
- device = 'cuda'
+ device = accelerator.device
  model.to(device)
```

### Preparing your objects

Next, you need to pass all of the important objects related to training into [`~Accelerator.prepare`]. 🤗 Accelerate will
make sure everything is setup in the current environment for you to start training:

```
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
```
These objects are returned in the same order they were sent in. By default when using `device_placement=True`, all of the objects that can be sent to the right device will be.
If you need to work with data that isn't passed to [~Accelerator.prepare] but should be on the active device, you should pass in the `device` you made earlier. 

<Tip warning={true}>

    Accelerate will only prepare objects that inherit from their respective PyTorch classes (such as `torch.optim.Optimizer`).

</Tip>

### Modifying the training loop

Finally, three lines of code need to be changed in the training loop. 🤗 Accelerate's DataLoader classes will automatically handle the device placement by default,
and [`~Accelerator.backward`] should be used for performing the backward pass:

```diff
-   inputs = inputs.to(device)
-   targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

With that, your training loop is now ready to use 🤗 Accelerate!

## The finished code

Below is the final version of the converted code: 

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## More Resources

To check out more ways on how to migrate to 🤗 Accelerate, check out our [interactive migration tutorial](https://huggingface.co/docs/accelerate/usage_guides/explore) which showcases other items that need to be watched for when using Accelerate and how to do so quickly.