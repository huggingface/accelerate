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

# Gradient synchronization

PyTorch's distributed module operates by communicating back and forth between all of the GPUs in your system.
This communication takes time, and ensuring all processes know the states of each other happens at particular triggerpoints
when using the `ddp` module. 

These triggerpoints are added to the PyTorch model, specifically their `forward()` and `backward()` methods. 
This happens when the model is wrapped with `DistributedDataParallel`:
```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

model = nn.Linear(10, 10)
ddp_model = DistributedDataParallel(model)
```
In Accelerate this conversion happens automatically when calling [`~Accelerator.prepare`] and passing in your model.

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()
  import torch.nn as nn
- from torch.nn.parallel import DistributedDataParallel

  model = nn.Linear(10,10)
+ model = accelerator.prepare(model)
```

## The slowdown in gradient accumulation

You now understand that PyTorch adds hooks to the `forward` and `backward` method of your PyTorch model when 
training in a distributed setup. But how does this risk slowing down your code?

In DDP (distributed data parallel), the specific order in which processes are performed and ran are expected
at specific points and these must also occur at roughly the same time before moving on.

The most direct example is when you update model parameters through
`optimizer.step()`.
Without gradient accumulation, all instances of the model need to have updated
their gradients computed, collated, and updated before moving on to the next
batch of data.
When performing gradient accumulation, you accumulate `n` loss gradients and
skip `optimizer.step()` until `n` batches have been reached. As all training
processes only need to synchronize by the time `optimizer.step()` is called,
without any modification to your training step, this needless inter-process
communication can cause a significant slowdown.

 How can you avoid this overhead?

## Solving the slowdown problem

Since you are skipping model parameter updates when training on these batches, their gradients do not need to be synchronized until the point where `optimizer.step()` is actually called. 
PyTorch cannot automagically tell when you need to do this, but they do provide a tool to help through the [`no_sync`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) context manager
that is added to your model after converting it to DDP.

Under this context manager, PyTorch will skip synchronizing the gradients when
`.backward()` is called, and the first call to `.backward()` outside this 
context manager will trigger the synchronization. See an example below:
```python
ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

for index, batch in enumerate(dataloader):
    inputs, targets = batch
    # Trigger gradient synchronization on the last batch
    if index != (len(dataloader) - 1):
        with ddp_model.no_sync():
            # Gradients only accumulate
            outputs = ddp_model(inputs)
            loss = loss_func(outputs)
            accelerator.backward(loss)
    else:
        # Gradients finally sync
        outputs = ddp_model(inputs)
        loss = loss_func(outputs)
        accelerator.backward(loss)
        optimizer.step()
```

In Accelerate to make this an API that can be called no matter the training device (though it may not do anything if you are not in a distributed system!),
`ddp_model.no_sync` gets replaced with [`~Accelerator.no_sync`] and operates the same way:

```diff
  ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

  for index, batch in enumerate(dataloader):
      inputs, targets = batch
      # Trigger gradient synchronization on the last batch
      if index != (len(dataloader)-1):
-         with ddp_model.no_sync():
+         with accelerator.no_sync(model):
              # Gradients only accumulate
              outputs = ddp_model(inputs)
              loss = loss_func(outputs, targets)
              accelerator.backward(loss)
      else:
          # Gradients finally sync
          outputs = ddp_model(inputs)
          loss = loss_func(outputs)
          accelerator.backward(loss)
          optimizer.step()
          optimizer.zero_grad()
```

As you may expect, the [`~Accelerator.accumulate`] function wraps around this conditional check by keeping track of the current batch number, leaving you with the final
gradient accumulation API:

```python
ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

for batch in dataloader:
    with accelerator.accumulate(model):
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

As a result, you should either use *`accelerator.accumulate` or `accelerator.no_sync`* when it comes to API choice. 

## Just how much of a slowdown is there, and easy mistakes you can make

To set up a realistic example, consider the following setup:

* Two single-GPU T4 nodes and one node with two GPUs
* Each GPU is a T4, and are hosted on GCP
* The script used is a modification of the [NLP Example](https://github.com/muellerzr/timing_experiments/blob/main/baseline.py) script
* Batch size per GPU is 16, and gradients are accumulated every 4 steps

All scripts are available in [this repository](https://github.com/muellerzr/timing_experiments).

If not careful about gradient synchronization and GPU communication, a *large* amount of time can be wasted 
from when these GPUs communicate to each other during unnecessary periods.

By how much?

Reference:
- Baseline: uses no synchronization practices discussed here
- `no_sync` improperly: `no_sync` only around the `backward` call, not the `forward`
- `no_sync`: using the `no_sync` pattern properly
- `accumulate`: using [`~Accelerator.accumulate`] properly

Below are the average seconds per batch iterating over 29 batches of data for each setup on both a single node and on the dual-node setup:

|             | Baseline  | `no_sync` improperly | `no_sync` | `accumulate`| 
| :---------: | :-------: | :------------------: | :-------: | :---------: |
| Multi-Node  | 2±0.01s    | 2.13±0.08s | **0.91±0.11s** | **0.91±0.11s** |
| Single Node | 0.50±0.01s | 0.50±0.01s | **0.41±0.015s** | **0.41±0.015s** |

As you can see, if you are not careful about how you set up your gradient synchronization, you can get upwards of more than a 2x slowdown during training!

If you are worried about making sure everything is done properly, we highly recommend utilizing the [`~Accelerator.accumulate`] function and passing in
`gradient_accumulation_steps` or `gradient_accumulation_plugin` to the [`Accelerator`] object so Accelerate can handle this for you.

### `no_sync` requires additional GPU memory when using FSDP

Be aware that not syncing gradients can have adverse effects while performing FSDP training. As it has been warned in `torch`, the [`no_sync` context manager for FSDP](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync) will require additional memory.

Therefore in memory intensive situations while using FSDP, we recommend to set `sync_each_batch` to `True` in the [`~utils.GradientAccumulationPlugin`] to disable `no_sync`.

See the example below where we fine-tune Mixtral (47B parameters) on 8 A100-80GB GPUs. We see that even for a modest `gradient_accumulation_steps=2` we quickly go out-of-memory (OOM) if `no_sync` is enabled. Again, this is due to additional memory overheads due to FSDP's `no_sync`. However, if `no_sync` is disabled via `sync_each_batch=True`, then the memory consumption for `gradient_accumulation_steps=16` reverts to that of `gradient_accumulation_steps=1`.

| Model           | `no_sync` (accum=1) | `no_sync` (accum=2) | `no_sync` disabled (accum=16)
| :-------------: | :-----------------: | :-----------------: | :-----------------: 
mixtral 8x7B      | 69G                 | OOM                 | 69G

> [!WARNING] 
> Disabling `no_sync` means there _will be slowdown_ due the extra data syncs, as explained by the earlier sections of this guide.