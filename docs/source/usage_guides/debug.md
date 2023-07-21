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

# Debugging Distributed Operations

When running scripts in a distributed fashion, often functions such as [`Accelerator.gather`] and [`Accelerator.reduce`] (and others) are neccessary to grab tensors across devices and perform certain operations on them. However, if the tensors which are being grabbed are not the proper shapes then this will result in your code hanging forever. The only sign that exists of this truly happening is hitting a timeout exception from `torch.distributed`, but this can get quite costly as usually the timeout is 10 minutes.

Accelerate now has a `debug` mode which adds a neglible amount of time to each operation, but allows it to verify that the inputs you are bringing in can *actually* perform the operation you want **without** hitting this timeout problem!

## Visualizing the problem

To have a tangible example of this issue, let's take the following setup (on 2 GPUs):

```python
from accelerate import PartialState

state = PartialState()
if state.process_index == 0:
    tensor = torch.tensor([[0.0, 1, 2, 3, 4]]).to(state.device)
else:
    tensor = torch.tensor([[[0.0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]).to(state.device)

broadcast_tensor = broadcast(tensor)
print(broadcast_tensor)
```

We've created a single tensor on each device, with two radically different shapes. With this setup if we want to perform an operation such as [`utils.broadcast`], we would forever hit a timeout because `torch.distributed` requires that these operations have the **exact same shape** across all processes for it to work.

If you run this yourself, you will find that `broadcast_tensor` can be printed on the main process, but its results won't quite be right, and then it will just hang never printing it on any of the other processes:

```
>>> tensor([[0, 1, 2, 3, 4]], device='cuda:0')
```

## The solution

By enabling Accelerate's operational debug mode, Accelerate will properly find and catch errors such as this and provide a very clear traceback immediatly: 

```
Traceback (most recent call last):
  File "/home/zach_mueller_huggingface_co/test.py", line 18, in <module>
    main()
  File "/home/zach_mueller_huggingface_co/test.py", line 15, in main
        main()broadcast_tensor = broadcast(tensor)
  File "/home/zach_mueller_huggingface_co/accelerate/src/accelerate/utils/operations.py", line 303, in wrapper
    broadcast_tensor = broadcast(tensor)
accelerate.utils.operations.DistributedOperationException: Cannot apply desired operation due to shape mismatches. All shapes across devices must be valid.

Operation: `accelerate.utils.operations.broadcast`
Input shapes:
  - Process 0: [1, 5]
  - Process 1: [1, 2, 5]
```

This explains that the shapes across our devices were *not* the same, and that we should ensure that they match properly to be compatible. Typically this means that there is either an extra dimension, or certain dimensions are incompatible with the operation.

To enable this please do one of the following:

Enable it through the questionarre during `accelerate config` (recommended)

From the CLI: 

```
accelerate launch --debug {my_script.py} --arg1 --arg2
```

As an environmental variable (which avoids the need for `accelerate launch`):

```
ACCELERATE_DEBUG_MODE="1" accelerate launch {my_script.py} --arg1 --arg2
```

Manually changing the `config.yaml` file:

```diff
 compute_environment: LOCAL_MACHINE
+debug: true
```



