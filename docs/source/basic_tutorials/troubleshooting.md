<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Troubleshoot

This guide provides solutions to some issues you might encounter when using Accelerate. Not all errors are covered because Accelerate is an active library that is continuously evolving and there are many different use cases and distributed training setups. If the solutions described here don't help with your specific error, please take a look at the [Ask for help](#ask-for-help) section to learn where and how to get help.

## Logging

Logging can help you identify where an error is coming from. In a distributed setup with multiple processes, logging can be a challenge, but Accelerate provides the [`~accelerate.logging`] utility to ensure logs are synchronized.

To troubleshoot an issue, use [`~accelerate.logging`] instead of the standard Python [`logging`](https://docs.python.org/3/library/logging.html#module-logging) module. Set the verbosity level (`INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`) with the `log_level` parameter, and then you can either:

1. Export the `log_level` as the `ACCELERATE_LOG_LEVEL` environment variable.
2. Pass the `log_level` directly to `get_logger`.

For example, to set `log_level="INFO"`:

```py
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")
```

By default, the log is called on main processes only. To call it on all processes, pass `main_process_only=False`.
If a log should be called on all processes and in order, also pass `in_order=True`.

```py
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")
# log all processes
logger.debug("thing_to_log", main_process_only=False)
# log all processes in order
logger.debug("thing_to_log", main_process_only=False, in_order=True)
```

## Hanging code and timeout errors

There can be many reasons why your code is hanging. Let's take a look at how to solve some of the most common issues that can cause your code to hang.

### Mismatched tensor shapes

Mismatched tensor shapes is a common issue that can cause your code to hang for a significant amount of time on a distributed setup.

When running scripts in a distributed setup, functions such as [`Accelerator.gather`] and [`Accelerator.reduce`] are necessary to grab tensors across devices to collectively perform operations on them. These (and other) functions rely on `torch.distributed` to perform a `gather` operation, which requires tensors to have the **exact same shape** across all processes. When the tensor shapes don't match, your code hangs and you'll eventually hit a timeout exception.

You can use Accelerate's operational debug mode to immediately catch this issue. We recommend enabling this mode during the `accelerate config` setup, but you can also enable it from the CLI, as an environment variable, or by manually editing the `config.yaml` file.

<hfoptions id="mismatch">
<hfoption id="CLI">

```bash
accelerate launch --debug {my_script.py} --arg1 --arg2
```

</hfoption>
<hfoption id="environment variable">

If enabling debug mode as an environment variable, you don't need to call `accelerate launch`.

```bash
ACCELERATE_DEBUG_MODE="1" torchrun {my_script.py} --arg1 --arg2
```

</hfoption>
<hfoption id="config.yaml">

Add `debug: true` to your `config.yaml` file.

```yaml
compute_environment: LOCAL_MACHINE
debug: true
```

</hfoption>
</hfoptions>

Once you enable debug mode, you should get a traceback that points to the tensor shape mismatch issue.

```py
Traceback (most recent call last):
  File "/home/zach_mueller_huggingface_co/test.py", line 18, in <module>
    main()
  File "/home/zach_mueller_huggingface_co/test.py", line 15, in main
    broadcast_tensor = broadcast(tensor)
  File "/home/zach_mueller_huggingface_co/accelerate/src/accelerate/utils/operations.py", line 303, in wrapper
accelerate.utils.operations.DistributedOperationException:

Cannot apply desired operation due to shape mismatches. All shapes across devices must be valid.

Operation: `accelerate.utils.operations.broadcast`
Input shapes:
  - Process 0: [1, 5]
  - Process 1: [1, 2, 5]
```

### Early stopping

For early stopping in distributed training, if each process has a specific stopping condition (e.g. validation loss), it may not be synchronized across all processes. As a result, a break can happen on process 0 but not on process 1 which will cause your code to hang indefinitely until a timeout occurs.

If you have early stopping conditionals, use the `set_trigger` and `check_trigger` methods to make sure all the processes
are ended correctly.

```py
# Assume `should_do_breakpoint` is a custom defined function that returns a conditional, 
# and that conditional might be true only on process 1
if should_do_breakpoint(loss):
    accelerator.set_trigger()

# Later in the training script when we need to check for the breakpoint
if accelerator.check_trigger():
    break
```

### Low kernel versions on Linux

On Linux with kernel version < 5.5, hanging processes have been reported. To avoid this problem, upgrade your system to a later kernel version.

### MPI

If your distributed CPU training job using MPI is hanging, ensure that you have
[passwordless SSH](https://www.open-mpi.org/faq/?category=rsh#ssh-keys) setup (using keys) between the nodes. This means
that for all nodes in your hostfile, you should to be able to SSH from one node to another without being prompted for a password.

Next, try to run the `mpirun` command as a sanity check. For example, the command below should print out the
hostnames for each of the nodes.

```bash
mpirun -f hostfile -n {number of nodes} -ppn 1 hostname
```

## Out-of-Memory

One of the most frustrating errors when it comes to running training scripts is hitting "Out-of-Memory" on devices like CUDA, XPU or CPU. The entire script needs to be restarted and any progress is lost.

To address this problem, Accelerate provides the [`find_executable_batch_size`] utility that is heavily based on [toma](https://github.com/BlackHC/toma).
This utility retries code that fails due to OOM (out-of-memory) conditions and automatically lowers batch sizes. For each OOM condition, the algorithm decreases the batch size by half and retries the code until it succeeds.

To use [`find_executable_batch_size`], restructure your training function to include an inner function with `find_executable_batch_size` and build your dataloaders inside it. At a minimum, this only takes 4 new lines of code.

<Tip warning={true}> 

The inner function **must** take batch size as the first parameter, but we do not pass one to it when called. The wrapper will handle this for you. Any object (models, optimizers) that consumes device memory and is passed to the [`Accelerator`] also **must** be declared inside the inner function.

</Tip>

```diff
def training_function(args):
    accelerator = Accelerator()

+   @find_executable_batch_size(starting_batch_size=args.batch_size)
+   def inner_training_loop(batch_size):
+       nonlocal accelerator # Ensure they can be used in our context
+       accelerator.free_memory() # Free all lingering references
        model = get_model()
        model.to(accelerator.device)
        optimizer = get_optimizer()
        train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
        lr_scheduler = get_scheduler(
            optimizer, 
            num_training_steps=len(train_dataloader)*num_epochs
        )
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        train(model, optimizer, train_dataloader, lr_scheduler)
        validate(model, eval_dataloader)
+   inner_training_loop()
```

## Non-reproducible results between device setups

If you changed the device setup and observe different model performance, it is likely you didn't update your script when moving from one setup to another. Even if you're using the same script with the same batch size, the results will still be different on a TPU, multi-GPU, and single GPU.

For example, if you were training on a single GPU with a batch size of 16 and you move to a dual GPU setup, you need to change the batch size to 8 to have the same effective batch size. This is because when training with Accelerate, the batch size passed to the dataloader is the **batch size per GPU**.

To make sure you can reproduce the results between the setups, make sure to use the same seed, adjust the batch size accordingly, and consider scaling the learning rate.

For more details and a quick reference for batch sizes, check out the [Comparing performance between different device setups](../concept_guides/performance) guide.

## Performance issues on different GPUs

If your multi-GPU setup consists of different GPUs, you may encounter some performance issues:

- There may be an imbalance in GPU memory between the GPUs. In this case, the GPU with the smaller memory will limit the batch size or the size of the model that can be loaded onto the GPUs.
- If you are using GPUs with different performance profiles, the performance will be driven by the slowest GPU you are using because the other GPUs will have to wait for it to complete its workload.

Vastly different GPUs within the same setup can lead to performance bottlenecks.

## Ask for help

If none of the solutions and advice here helped resolve your issue, you can always reach out to the community and Accelerate team for help.

- Ask for help on the Hugging Face forums by posting your question in the [Accelerate category](https://discuss.huggingface.co/c/accelerate/18). Make sure to write a descriptive post with relevant context about your setup and reproducible code to maximize the likelihood that your problem is solved!

- Post a question on [Discord](http://hf.co/join/discord), and let the team and the community help you.

- Create an Issue on the Accelerate [GitHub repository](https://github.com/huggingface/accelerate/issues) if you think you've found a bug related to the library. Include context regarding the bug and details about your distributed setup to help us better figure out what's wrong and how we can fix it.
