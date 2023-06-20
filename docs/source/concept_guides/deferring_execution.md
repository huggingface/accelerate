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

# Deferring Executions

When you run your usual script, instructions are executed in order. Using 🤗 Accelerate to deploy your script on several
GPUs at the same time introduces a complication: while each process executes all instructions in order, some may be
faster than others.

You might need to wait for all processes to have reached a certain point before executing a given instruction. For
instance, you shouldn't save a model before being sure every process is done with training, and you wouldn't want to 
continue training before all the model weights have been loaded in. To do this, just write the following line in your code:

```
accelerator.wait_for_everyone()
```

This instruction will block all the processes that arrive first until all the other processes have reached that
point (if you run your script on just one GPU or CPU, this won't do anything).

A few example cases of when to use this utility are listed below:

<Tip>

    Some of these are utilized with the [`~Accelerator.main_process_first`] context manager, which utilizes [`~Accelerator.wait_for_everyone`] to 
    run a particular set of code on the main process beforehand before triggering and launching the other processes

</Tip>

## Downloading a Dataset 

When downloading a dataset, you should download it first on the main process and then load the cached dataset afterward

<Tip>

    `load_dataset` will perform a lock under the hood to stop multiple downloads from happening at once, but if you are downloading something 
    not using this library you should use this method.
    
</Tip>

```python
with accelerator.main_process_first():
    datasets = load_dataset("glue", "mrpc")
```

Under the hood this is the same as calling: 

```python
# First do something on the main process
if accelerator.is_main_process:
    datasets = load_dataset("glue", "mrpc")
else:
    accelerator.wait_for_everyone()

# And then send it to the rest of them
if not accelerator.is_main_process:
    datasets = load_dataset("glue", "mrpc")
else:
    accelerator.wait_for_everyone()
```

## Saving the `state_dict`

When saving the `state_dict` of the model, since you would normally save one file on just the main process
you should specify that:

```python
if accelerator.is_main_process:
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), "weights.pth")
```

## Loading in the `state_dict`

When loading in the `state_dict` to a model, optimizer, or scheduler, you should wait 
for all workers to have the weights loaded in before moving on to training

```python
with accelerator.main_process_first():
    state = torch.load("weights.pth")
    model.load_state_dict(state)
```

## Applying a multi-worker CPU operation 

Applying a `map()` operation on multiple workers, such as tokenizing should be done on the 
main process first, and then propagated to each one. 

```python
datasets = load_dataset("glue", "mrpc")

with accelerator.main_process_first():
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
```
