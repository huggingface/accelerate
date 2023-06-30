<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Accelerator

The [`Accelerator`] is the main class provided by 🤗 Accelerate. 
It serves at the main entry point for the API. 

## Quick adaptation of your code

To quickly adapt your script to work on any kind of setup with 🤗 Accelerate just:

1. Initialize an [`Accelerator`] object (that we will call `accelerator` throughout this page) as early as possible in your script.
2. Pass your dataloader(s), model(s), optimizer(s), and scheduler(s) to the [`~Accelerator.prepare`] method.
3. Remove all the `.cuda()` or `.to(device)` from your code and let the `accelerator` handle the device placement for you. 

<Tip>

    Step three is optional, but considered a best practice.

</Tip>

4. Replace `loss.backward()` in your code with `accelerator.backward(loss)`
5. Gather your predictions and labels before storing them or using them for metric computation using [`~Accelerator.gather`]

<Tip warning={true}>

    Step five is mandatory when using distributed evaluation
    
</Tip>

In most cases this is all that is needed. The next section lists a few more advanced use cases and nice features
you should search for and replace by the corresponding methods of your `accelerator`:

## Advanced recommendations

### Printing

`print` statements should be replaced by [`~Accelerator.print`] to be printed once per process:

```diff
- print("My thing I want to print!")
+ accelerator.print("My thing I want to print!")
```

### Executing processes

#### Once on a single server

For statements that should be executed once per server, use [`~Accelerator.is_local_main_process`]:

```python
if accelerator.is_local_main_process:
    do_thing_once_per_server()
```

A function can be wrapped using the [`~Accelerator.on_local_main_process`] function to achieve the same 
behavior on a function's execution:

```python
@accelerator.on_local_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once_per_server()
```

#### Only ever once across all servers

For statements that should only ever be executed once, use [`~Accelerator.is_main_process`]:

```python
if accelerator.is_main_process:
    do_thing_once()
```

A function can be wrapped using the [`~Accelerator.on_main_process`] function to achieve the same 
behavior on a function's execution:

```python
@accelerator.on_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once()
```

#### On specific processes

If a function should be ran on a specific overall or local process index, there are similar decorators 
to achieve this:

```python
@accelerator.on_local_process(local_process_idx=0)
def do_my_thing():
    "Something done on process index 0 on each server"
    do_thing_on_index_zero_on_each_server()
```

```python
@accelerator.on_process(process_index=0)
def do_my_thing():
    "Something done on process index 0"
    do_thing_on_index_zero()
```

### Synchronicity control

Use [`~Accelerator.wait_for_everyone`] to make sure all processes join that point before continuing. (Useful before a model save for instance).

### Saving and loading

```python
model = MyModel()
model = accelerator.prepare(model)
```

Use [`~Accelerator.save_model`] instead of `torch.save` to save a model. It will remove all model wrappers added during the distributed process, get the state_dict of the model and save it.

```diff
- torch.save(state_dict, "my_state.pkl")
+ accelerator.save_model(model, save_directory)
```

[`~Accelerator.save_model`] can also save a model into sharded checkpoints or with safetensors format.
Here is an example: 

```python
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

#### 🤗 Transformers models

If you are using models from the [🤗 Transformers](https://huggingface.co/docs/transformers/) library, you can use the `.save_pretrained()` method.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
model = accelerator.prepare(model)

# ...fine-tune with PyTorch...

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
```

This will ensure your model stays compatible with other 🤗 Transformers functionality like the `.from_pretrained()` method.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

### Operations

Use [`~Accelerator.clip_grad_norm_`] instead of ``torch.nn.utils.clip_grad_norm_`` and [`~Accelerator.clip_grad_value_`] instead of ``torch.nn.utils.clip_grad_value``

### Gradient Accumulation

To perform gradient accumulation use [`~Accelerator.accumulate`] and specify a gradient_accumulation_steps. 
This will also automatically ensure the gradients are synced or unsynced when on 
multi-device training, check if the step should actually be performed, and auto-scale the loss:

```diff
- accelerator = Accelerator()
+ accelerator = Accelerator(gradient_accumulation_steps=2)

  for (input, label) in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, labels)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```
#### GradientAccumulationPlugin
[[autodoc]] utils.GradientAccumulationPlugin


Instead of passing `gradient_accumulation_steps` you can instantiate a GradientAccumulationPlugin and pass it to the [`Accelerator`]'s `__init__`
as `gradient_accumulation_plugin`. You can only pass either one of `gradient_accumulation_plugin` or `gradient_accumulation_steps` passing both will raise an error.
```diff
from accelerate.utils import GradientAccumulationPlugin

gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2)
- accelerator = Accelerator()
+ accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin)
```

In addition to the number of steps, this also lets you configure whether or not you adjust your learning rate scheduler to account for the change in steps due to accumulation.

## Overall API documentation:

[[autodoc]] Accelerator
