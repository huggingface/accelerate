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

# Performing gradient accumulation with Accelerate

Gradient accumulation is a technique where you can train on bigger batch sizes than 
your machine would normally be able to fit into memory. This is done by accumulating gradients over
several batches, and only stepping the optimizer after a certain number of batches have been performed.

While technically standard gradient accumulation code would work fine in a distributed setup, it is not the most efficient
method for doing so and you may experience considerable slowdowns!

In this tutorial you will see how to quickly setup gradient accumulation and perform it with the utilities provided in Accelerate,
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

## Converting it to Accelerate

First the code shown earlier will be converted to utilize Accelerate without the special gradient accumulation helper:

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

## Letting Accelerate handle gradient accumulation

All that is left now is to let Accelerate handle the gradient accumulation for us. To do so you should pass in a `gradient_accumulation_steps` parameter to [`Accelerator`], dictating the number 
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
training on. Accelerate automagically does this for you by default. Behind the scenes we instantiate a [`GradientAccumulationPlugin`] configured to do this.

</Tip>

<Tip warning={true}>

The [`state.GradientState`] is sync'd with the active dataloader being iterated upon. As such it assumes naively that when we have reached the end of the dataloader everything will sync and a step will be performed. To disable this, set `sync_with_dataloader` to be `False` in the [`GradientAccumulationPlugin`]:

```{python}
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

plugin = GradientAccumulationPlugin(sync_with_dataloader=False)
accelerator = Accelerator(..., gradient_accumulation_plugin=plugin)
```

</Tip>

## The finished code

Below is the finished implementation for performing gradient accumulation with Accelerate

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

Here is a self-contained example that you can run to see gradient accumulation in action with Accelerate:

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
class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1, 1)))

    def forward(self, inputs):
        return inputs @ self.weight

model = SimpleLinearModel()
model_clone = copy.deepcopy(model)
criterion = torch.nn.MSELoss()
model_optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, model_optimizer, dataloader = accelerator.prepare(model, model_optimizer, dataloader)
model_clone_optimizer = torch.optim.SGD(model_clone.parameters(), lr=0.02)
print(f"initial model weight is {model.weight.mean().item():.5f}")
print(f"initial model weight is {model_clone.weight.mean().item():.5f}")
for i, (inputs, labels) in enumerate(dataloader):
    with accelerator.accumulate(model):
        inputs = inputs.view(-1, 1)
        print(i, inputs.flatten())
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        model_optimizer.step()
        model_optimizer.zero_grad()
loss = criterion(x.view(-1, 1) @ model_clone.weight, y.view(-1, 1))
model_clone_optimizer.zero_grad()
loss.backward()
model_clone_optimizer.step()
print(f"w/ accumulation, the final model weight is {model.weight.mean().item():.5f}")
print(f"w/o accumulation, the final model weight is {model_clone.weight.mean().item():.5f}")
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

## Gradient accumulation on training samples of variable size

As was pointed out in this [blog-post](https://huggingface.co/blog/gradient_accumulation), which points out a common error that occurs when perfoming gradient accumulation on training samples of variable size:

>  [...] for gradient accumulation across token-level tasks like causal LM training, the correct loss should be computed by the **total loss across all batches in a gradient accumulation step** divided by the **total number of all non padding tokens in those batches**. This is not the same as the average of the per-batch loss values. 

In other words, some adjustements must be made on losses that operate on a token-level basis.

### Skeleton code

```python
from accelerate import Accelerator
import math
gradient_accumulation_steps = 2
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

training_iterator = iter(training_dataloader)
num_samples_in_epoch = len(training_dataloader)
remainder = num_samples_in_epoch % gradient_accumulation_steps
remainder = remainder if remainder != 0 else gradient_accumulation_steps
total_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)
        

total_batched_samples = 0
for update_step in range(total_updates):
        # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
        # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
        batch_samples = []
        num_batches_in_step = gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
        for _ in range(num_batches_in_step):
            batch_samples += [next(training_iterator)]
            
        # get local num items in batch 
        num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
        num_items_in_batch = accelerator.gather_for_metrics(num_items_in_batch).sum().item()
            
        for batch in batch_samples:
            total_batched_samples += 1

            with accelerator.accumulate(model):
                # Since we performed prefetching, we need to manually set sync_gradients
                if total_batched_samples % gradient_accumulation_steps != 0:
                    accelerator.gradient_state._set_sync_gradients(False)
                else:
                    accelerator.gradient_state._set_sync_gradients(True)

                inputs, targets = batch
                outputs = model(inputs)
                loss = loss_function(outputs, targets) # the loss function shoud sum over samples rather than averaging
                
                # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps
                loss = (loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
```

### Self-contained causal LM example

```py
import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import  get_logger
from torch.utils.data import Dataset, DataLoader
import math

# seed
set_seed(0)
logger = get_logger(__name__)

class MyDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.len = num_samples

    def __getitem__(self, index):
        input_ids = torch.arange(1, index+2, dtype=torch.float32)
        labels = torch.remainder(input_ids, 2)
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return self.len
    
def collate_fn(features):
    input_ids = torch.nn.utils.rnn.pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=-100)
    labels = torch.nn.utils.rnn.pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids[..., None], "labels": labels[..., None]}

# define toy inputs and labels
gradient_accumulation_steps = 2
batch_size = 4

# define accelerator
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

# define dataset and dataloader
# for this toy example, we'll compute gradient descent over one single global batch
dataset = MyDataset(batch_size*gradient_accumulation_steps*accelerator.num_processes)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# define model, model_optimizer and loss function
model = torch.nn.Linear(1, 2, bias=False)
model_clone = copy.deepcopy(model)
criterion = torch.nn.CrossEntropyLoss(reduction="sum") # must sum over samples rather than averaging
model_optimizer = torch.optim.SGD(model.parameters(), lr=0.08)


logger.warning(f"initial model weight is {model.weight.detach().cpu().squeeze()}", main_process_only=True)
logger.warning(f"initial model clone weight is {model_clone.weight.detach().cpu().squeeze()}", main_process_only=True)

# prepare artifacts - accelerator handles device placement and dataloader splitting
model, model_optimizer = accelerator.prepare(model, model_optimizer)
dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
training_iterator = iter(dataloader)

num_samples_in_epoch = len(dataloader)
remainder = num_samples_in_epoch % gradient_accumulation_steps
remainder = remainder if remainder != 0 else gradient_accumulation_steps
total_gradient_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)

total_batched_samples = 0
for update_step in range(total_gradient_updates):
        # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
        # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
        batch_samples = []
        num_batches_in_step = gradient_accumulation_steps if update_step != (total_gradient_updates - 1) else remainder
        for _ in range(num_batches_in_step):
            batch_samples += [next(training_iterator)]
            
        # get local num items in batch 
        local_num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        logger.warning(f"Step {update_step} - Device {accelerator.process_index} - num items in the local batch {local_num_items_in_batch}", main_process_only=False)

        # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
        num_items_in_batch = accelerator.gather_for_metrics(local_num_items_in_batch).sum().item()
        logger.warning(f"Total num items {num_items_in_batch}", main_process_only=True)

        for batch in batch_samples:
            inputs, labels = batch["input_ids"], batch["labels"]
            total_batched_samples += 1

            with accelerator.accumulate(model):
                # Since we performed prefetching, we need to manually set sync_gradients
                if total_batched_samples % gradient_accumulation_steps != 0:
                    accelerator.gradient_state._set_sync_gradients(False)
                else:
                    accelerator.gradient_state._set_sync_gradients(True)

                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 2), labels.view(-1).to(torch.int64))
                
                # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps 
                loss = (loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch
                accelerator.backward(loss)
                model_optimizer.step()
                model_optimizer.zero_grad()
                

logger.warning(f"Device {accelerator.process_index} - w/ accumulation, the final model weight is {accelerator.unwrap_model(model).weight.detach().cpu().squeeze()}", main_process_only=False)

# We know do the same operation but on a single device and without gradient accumulation

if accelerator.is_main_process:
    # prepare one single entire batch
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
    full_batch_without_accum = next(iter(dataloader))
    total_inputs, total_labels = full_batch_without_accum["input_ids"], full_batch_without_accum["labels"]
    model_clone_optimizer = torch.optim.SGD(model_clone.parameters(), lr=0.08)
    
    # train the cloned model
    loss = torch.nn.CrossEntropyLoss(reduction="mean")(model_clone(total_inputs).view(-1, 2), total_labels.view(-1).to(torch.int64))
    model_clone_optimizer.zero_grad()
    loss.backward()
    model_clone_optimizer.step()
    
    # We should have the same final weights.
    logger.warning(f"w/o accumulation, the final model weight is {model_clone.weight.detach().cpu().squeeze()}")

```

Results on a single device:
```
initial model weight is tensor([-0.0075,  0.5364])
initial model clone weight is tensor([-0.0075,  0.5364])
Step 0 - Device 0 - num items in the local batch 36
Total num items 36
Device 0 - w/ accumulation, the final model weight is tensor([0.0953, 0.4337])
w/o accumulation, the final model weight is tensor([0.0953, 0.4337])
```

Results on a two devices set-up:
```
initial model weight is tensor([-0.0075,  0.5364])
initial model clone weight is tensor([-0.0075,  0.5364])
Step 0 - Device 0 - num items in the local batch 52
Step 0 - Device 1 - num items in the local batch 84
Total num items 136
Device 1 - w/ accumulation, the final model weight is tensor([0.2117, 0.3172])
Device 0 - w/ accumulation, the final model weight is tensor([0.2117, 0.3172])
w/o accumulation, the final model weight is tensor([0.2117, 0.3172])
```