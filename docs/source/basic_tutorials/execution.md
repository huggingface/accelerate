<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Execution process

When working with distributed training systems, it is important to manage how and when processes are executed across GPUs. Some processes are completed faster than others, and some processes shouldn't begin if others haven't finished yet. Accelerate provides tools for orchestrating when processes are executed to ensure everything remains synchronized across all devices.

This tutorial will teach you how to execute a process on only one machine and how to delay execution until all processes have reached a certain point.

## Execute on one process

Certain code only needs to be run once on a given machine, such as printing a log statement or only displaying one progress bar on the local main process.

<hfoptions id="local-execution">
<hfoption id="statements">

You should use `accelerator.is_local_main_process` to indicate code that should only be executed once.

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
```

You could also wrap a statement with `accelerator.is_local_main_process`.

> [!TIP]
> For standalone `print` statements that aren't wrapped in `accelerator.is_local_main_process`, replace `print` with Accelerate's [`~Accelerator.print`] method to only print once per process.

```py
if accelerator.is_local_main_process:
    print("Accelerate is the best")
```

</hfoption>
<hfoption id="function">

For a function that should only be executed once, use [`~Accelerator.on_local_main_process`].

```py
@accelerator.on_local_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once_per_server()
```

</hfoption>
</hfoptions>

You could also direct Accelerate to execute code once across *all processes* regardless of the number of machines. This is useful if you're uploading a final model to the Hub.

<hfoptions id="main-execution">
<hfoption id="statement">

You should use `accelerator.is_main_process` to indicate code that should only be executed once across all processes.

```py
if accelerator.is_main_process:
    repo.push_to_hub()
```

</hfoption>
<hfoption id="function">

For a function that should only be executed once across all processes, use [`~Accelerator.on_main_process`].

```py
@accelerator.on_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once()
```

</hfoption>
</hfoptions>

## Execute on a specific process

Accelerate can also help you execute functions that should only be executed on a specific process or a local process index.

<hfoptions id="specific-execution">
<hfoption id="specific process">

Use the [`~Accelerator.on_process`] method and specify the process index to execute a function on.

```py
@accelerator.on_process(process_index=0)
def do_my_thing():
    "Something done on process index 0"
    do_thing_on_index_zero()
```

</hfoption>
<hfoption id="local process">

Use the [`~Accelerator.on_local_process`] method and specify the local process index to execute a function on.

```py
@accelerator.on_local_process(local_process_idx=0)
def do_my_thing():
    "Something done on process index 0 on each server"
    do_thing_on_index_zero_on_each_server()
```

</hfoption>
</hfoptions>

## Defer execution

When you run your script on several GPUs at the same time, some code may be executed faster than others. You might need to wait for all processes to reach a certain point before executing the next set of instructions. For instance, you shouldn’t save a model before making sure every process is done with training.

To do this, add [`~Accelerator.wait_for_everyone`] in your code. This blocks all processes that have finished first from continuing until all remaining processes have reached the same point (this has no effect if you're running on a single GPU or CPU).

```py
accelerator.wait_for_everyone()
```
