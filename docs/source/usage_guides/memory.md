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

# Memory Utilities

One of the most frustrating errors when it comes to running training scripts is hitting "CUDA Out-of-Memory", 
as the entire script needs to be restarted, progress is lost, and typically a developer would want to simply
start their script and let it run.

`Accelerate` provides a utility heavily based on [toma](https://github.com/BlackHC/toma) to give this capability.

## find_executable_batch_size

This algorithm operates with exponential decay, decreasing the batch size in half after each failed run on some 
training script. To use it, restructure your training function to include an inner function that includes this wrapper, 
and build your dataloaders inside it. At a minimum, this could look like 4 new lines of code. 
> Note: The inner function *must* take in the batch size as the first parameter, but we do not pass one to it when called. The wrapper handles this for us

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
