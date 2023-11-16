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

# Troubleshooting guide

This guide aims to provide you the tools and knowledge required to navigate some common issues. However, 
as 🤗 Accelerate continuously evolves and the use cases and setups are diverse, you might encounter an issue not covered in this 
guide. If the suggestions listed in this guide do not cover your such situation, please refer to the final section of 
the guide, [Asking for Help](#ask-for-help), to learn where to find help with your specific issue.

## Logging

When facing an error, logging can help narrow down where it is coming from. In a distributed setup with multiple processes, 
logging can be a challenge, but 🤗 Accelerate provides a utility that streamlines the logging process and ensures that 
logs are synchronized and managed effectively across the distributed setup. 

To troubleshoot an issue, use `accelerate.logging` instead of the standard Python `logging` module:

```diff
- import logging
+ from accelerate.logging import get_logger
- logger = logging.getLogger(__name__)
+ logger = get_logger(__name__)
```

To set the log level (`INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`), export it as the `ACCELERATE_LOG_LEVEL` environment,
or pass as `log_level` to `get_logger`:

```python
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")
```

By default, the log is called on main processes only. To call it on all processes, pass `main_process_only=False`.
If a log should be called on all processes and in order, also pass `in_order=True`.

## Hanging code and timeout errors

### Mismatched tensor shapes 

If your code seems to be hanging for a significant amount time on a distributed setup, a common cause is mismatched shapes of tensors on different 
devices. 

When running scripts in a distributed fashion, functions such as [`Accelerator.gather`] and [`Accelerator.reduce`] are 
necessary to grab tensors across devices to perform operations on them collectively. These (and other) functions rely on 
`torch.distributed` performing a `gather` operation, which requires that tensors have the **exact same shape** across all processes.
When the tensor shapes don't match, you will experience handing code, and eventually hit a timeout exception. 

If you suspect this to be the case, use Accelerate's operational debug mode to immediately catch the issue.

The recommended way to enable Accelerate's operational debug mode is during `accelerate config` setup. 
Alternative ways to enable debug mode are: 

* From the CLI: 

```bash
accelerate launch --debug {my_script.py} --arg1 --arg2
```

* As an environmental variable (which avoids the need for `accelerate launch`):

```bash
ACCELERATE_DEBUG_MODE="1" torchrun {my_script.py} --arg1 --arg2
```

* Manually changing the `config.yaml` file:

```diff
 compute_environment: LOCAL_MACHINE
+debug: true
```

Once you enable the debug mode, you should get a similar traceback that points to the tensor shape mismatch issue:

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

### Early stopping leads to hanging

When doing early stopping in distributed training, if each process has a specific stopping condition (e.g. validation loss), 
it may not be synchronized across all of them. As a result, a break can happen on process 0 but not on process 1. 
This will cause the code to hang indefinitely until a timeout occurs.

If you have early stopping conditionals, use `set_breakpoint` and `check_breakpoint` methods to make sure all the processes
are ended correctly:

```py
# Assume `should_do_breakpoint` is a custom defined function that returns a conditional, 
# and that conditional might be true only on process 1
if should_do_breakpoint(loss):
    accelerator.set_breakpoint()

# Later in the training script when we need to check for the breakpoint
if accelerator.check_breakpoint():
    break
```

### Hanging on low kernel versions on Linux

This is a known issue. On Linux with kernel version < 5.5, hanging processes have been reported. To avoid 
encountering this problem, we recommend upgrading your system to a later kernel version.

## CUDA out of memory

One of the most frustrating errors when it comes to running training scripts is hitting "CUDA Out-of-Memory", 
as the entire script needs to be restarted, progress is lost, and typically a developer would want to simply
start their script and let it run.

To address this problem, `Accelerate` offers a utility `find_executable_batch_size` that is heavily based on [toma](https://github.com/BlackHC/toma).
The utility retries code that fails due to OOM (out-of-memory) conditions and lowers batch sizes automatically.

### find_executable_batch_size

This algorithm operates with exponential decay, decreasing the batch size in half after each failed run on some 
training script. To use it, restructure your training function to include an inner function that includes this wrapper, 
and build your dataloaders inside it. At a minimum, this could look like 4 new lines of code. 

<Tip warning={true}> 

The inner function *must* take in the batch size as the first parameter, but we do not pass one to it when called. The wrapper handles this for us.

</Tip>

It should also be noted that anything which will consume CUDA memory and passed to the `accelerator` **must** be declared inside the inner function,
such as models and optimizers.

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

To find out more, check the documentation [here](../package_reference/utilities#accelerate.find_executable_batch_size).

## Non-reproducible results between device setups

If you have changed the device setup and are observing different model performance, this is likely due to the fact that 
you have not updated your script when moving from one setup to another. The same script with the same batch size across TPU, 
multi-GPU, and single-GPU with Accelerate will have different results. 

For example, if you were previously training on a single GPU with a batch size of 16, when moving to two GPU setup, 
you need to change the batch size to 8 to have the same effective batch size. This is because when training with Accelerate, 
the batch size passed to the dataloader is the **batch size per GPU**.

To make sure you can reproduce the results between the setups, make sure to use the same seed, adjust the batch size 
accordingly, consider scaling the learning rate.

For more details and a quick reference for batch sizes, check out the [Comparing performance between different device setups](../concept_guides/performance) guide.

## Performance issues on different GPUs

If your multi-GPU setup consists of different GPUs, you may hit some limitations:

- There may be an imbalance in GPU memory between the GPUs. In this case, the GPU with smaller memory will limit the batch size or the size of the model that can be loaded onto the GPUs.
- If you are using GPUs with different performance profiles, the performance will be driven by the slowest GPU that you are using as the other GPUs will have to wait for it to complete its workload.

Vastly different GPUs within the same setup can lead to performance bottlenecks.

## Ask for help

If the above troubleshooting tools and advice did not help you resolve your issue, reach out for help to the community 
and the team.

### Forums 

Ask for help on the Hugging Face forums - post your question in the [🤗Accelerate category](https://discuss.huggingface.co/c/accelerate/18) 
Make sure to write a descriptive post with relevant context about your setup and reproducible code to maximize the likelihood that your problem is solved!

### Discord

Post a question on [Discord](http://hf.co/join/discord), and let the team and the community help you.

### GitHub Issues

Create an Issue on the 🤗 Accelerate [GitHub repository](https://github.com/huggingface/accelerate/issues) if you suspect 
to have found a bug related to the library. Include context regarding the bug and details about your distributed setup
to help us better figure out what's wrong and how we can fix it.
