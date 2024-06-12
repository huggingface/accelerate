<!--
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# DDP Communication Hooks

DDP communication hooks provide a generic interface to control how gradients are communicated across workers by overriding the vanilla allreduce in `DistributedDataParallel`. A few built-in communication hooks are provided, and users can easily apply any of these hooks to optimize communication.

## Brief Descriptions of DDP Communication Hooks

- **FP16 Compression Hook**: Compresses gradients by casting them to half-precision floating-point format (`torch.float16`), reducing communication overhead.
- **BF16 Compression Hook**: Similar to FP16, but uses the Brain Floating Point format (`torch.bfloat16`), which can be more efficient on certain hardware.
- **PowerSGD Hook**: An advanced gradient compression algorithm that provides high compression rates and can accelerate bandwidth-bound distributed training.

In this tutorial, you will see how to quickly set up DDP communication hooks and perform training with the utilities provided in 🤗 Accelerate, which can be as simple as adding just one new line of code!

## Using DDP Communication Hooks with PyTorch's `DistributedDataParallel`

### Using FP16 Compression Hook with PyTorch

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[torch.cuda.current_device()])
model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Using BF16 Compression Hook with PyTorch

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[torch.cuda.current_device()])
model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Using PowerSGD Hook with PyTorch

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[torch.cuda.current_device()])
state = powerSGD_hook.PowerSGDState(process_group=None)
model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Converting it to 🤗 Accelerate

Now, let's see how to use the same hooks with the 🤗 Accelerate library.

### Using FP16 Compression Hook with 🤗 Accelerate

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### Using BF16 Compression Hook with 🤗 Accelerate

<Tip warning={true}>

BF16 Compression Hook API is experimental, and it requires NCCL version later than 2.9.6.

</Tip>

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### Using PowerSGD Hook with 🤗 Accelerate

<Tip warning={true}>

PowerSGD typically requires extra memory of the same size as the model’s gradients to enable error feedback, which can compensate for biased compressed communication and improve accuracy.

</Tip>

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.POWER_SGD)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## Additional Options for DDP Communication Hooks

### comm_wrapper

`comm_wrapper` is an option to wrap a communication hook with additional functionality. For example, it can be used to combine FP16 compression with other communication strategies. Currently supported wrappers are `no`, `fp16`, and `bf16`.

### comm_state_option

`comm_state_option` allows you to pass additional state information required by certain communication hooks. This is particularly useful for stateful hooks like `PowerSGD`, which require maintaining hyperparameters and internal states across training steps. Below is an example showcasing the use of `comm_state_option` with the `PowerSGD` hook.

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.POWER_SGD,
    comm_state_option={"matrix_approximation_rank": 2}
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

For more advanced usage and additional hooks, refer to the [PyTorch DDP Communication Hooks documentation](https://pytorch.org/docs/stable/ddp_comm_hooks.html).

This demonstrates how to use DDP communication hooks to optimize gradient communication in distributed training with the 🤗 Accelerate library.
