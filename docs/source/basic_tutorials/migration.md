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

# Add Accelerate to your code

Each distributed training framework has their own way of doing things which can require writing a lot of custom code to adapt it to your PyTorch training code and training environment. Accelerate offers a friendly way to interface with these distributed training frameworks without having to learn the specific details of each one. Accelerate takes care of those details for you, so you can focus on the training code and scale it to any distributed training environment.

In this tutorial, you'll learn how to adapt your existing PyTorch code with Accelerate and get you on your way toward training on distributed systems with ease! You'll start with a basic PyTorch training loop (it assumes all the training objects like `model` and `optimizer` have been setup already) and progressively integrate Accelerate into it.

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## Accelerator

The [`Accelerator`] is the main class for adapting your code to work with Accelerate. It knows about the distributed setup you're using such as the number of different processes and your hardware type. This class also provides access to many of the necessary methods for enabling your PyTorch code to work in any distributed training environment and for managing and executing processes across devices.

That's why you should always start by importing and creating an [`Accelerator`] instance in your script.

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

The [`Accelerator`] also knows which device to move your PyTorch objects to, so it is recommended to let Accelerate handle this for you.

```diff
- device = "cuda"
+ device = accelerator.device
  model.to(device)
```

## Prepare PyTorch objects

Next, you need to prepare your PyTorch objects (model, optimizer, scheduler, etc.) for distributed training. The [`~Accelerator.prepare`] method takes care of placing your model in the appropriate container (like single GPU or multi-GPU) for your training setup, adapting the optimizer and scheduler to use Accelerate's [`~optimizer.AcceleratedOptimizer`] and [`~scheduler.AcceleratedScheduler`], and creating a new dataloader that can be sharded across processes.

> [!TIP]
> Accelerate only prepares objects that inherit from their respective PyTorch classes such as `torch.optim.Optimizer`.

The PyTorch objects are returned in the same order they're sent.

```py
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
```

## Training loop

Finally, remove the `to(device)` calls to the inputs and targets in the training loop because Accelerate's DataLoader classes automatically places them on the right device. You should also replace the usual `backward()` pass with Accelerate's [`~Accelerator.backward`] method which scales the gradients for you and uses the appropriate `backward()` method depending on your distributed setup (for example, DeepSpeed or Megatron).

```diff
-   inputs = inputs.to(device)
-   targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

Put everything together and your new Accelerate training loop should now look like this!

```python
from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## Training features

Accelerate offers additional features - like gradient accumulation, gradient clipping, mixed precision training and more - you can add to your script to improve your training run. Let's explore these three features.

### Gradient accumulation

Gradient accumulation enables you to train on larger batch sizes by accumulating the gradients over multiple batches before updating the weights. This can be useful for getting around memory limitations. To enable this feature in Accelerate, specify the `gradient_accumulation_steps` parameter in the [`Accelerator`] class and add the [`~Accelerator.accumulate`] context manager to your script.

```diff
+ accelerator = Accelerator(gradient_accumulation_steps=2)
  model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

  for input, label in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, label)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```

### Gradient clipping

Gradient clipping is a technique to prevent "exploding gradients", and Accelerate offers:

* [`~Accelerator.clip_grad_value_`] to clip gradients to a minimum and maximum value
* [`~Accelerator.clip_grad_norm_`] for normalizing gradients to a certain value

### Mixed precision

Mixed precision accelerates training by using a lower precision data type like fp16 (half-precision) to calculate the gradients. For the best performance with Accelerate, the loss should be computed inside your model (like in Transformers models) because computations outside of the model are computed in full precision.

Set the mixed precision type to use in the [`Accelerator`], and then use the [`~Accelerator.autocast`] context manager to automatically cast the values to the specified data type.

> [!WARNING]
> Accelerate enables automatic mixed precision, so [`~Accelerator.autocast`] is only needed if there are other mixed precision operations besides those performed on loss by [`~Accelerator.backward`] which already handles the scaling.

```diff
+ accelerator = Accelerator(mixed_precision="fp16")
+ with accelerator.autocast():
      loss = complex_loss_function(outputs, target):
```

## Save and load

Accelerate can also save and load a *model* once training is complete or you can also save the model and optimizer *state* which could be useful for resuming training.

### Model

Once all processes are complete, unwrap the model with the [`~Accelerator.unwrap_model`] method before saving it because the [`~Accelerator.prepare`] method wrapped your model into the proper interface for distributed training. If you don't unwrap the model, saving the model state dictionary also saves any potential extra layers from the larger model and you won't be able to load the weights back into your base model.

You should use the [`~Accelerator.save_model`] method to unwrap and save the model state dictionary. This method can also save a model into sharded checkpoints or into the [safetensors](https://hf.co/docs/safetensors/index) format.

<hfoptions id="save">
<hfoption id="single checkpoint">

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```

<Tip>

For models from the [Transformers](https://hf.co/docs/transformers/index) library, save the model with the [`~transformers.PreTrainedModel.save_pretrained`] method so that it can be reloaded with the [`~transformers.PreTrainedModel.from_pretrained`] method.

```py
from transformers import AutoModel

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

</Tip>

To load your weights, use the [`~Accelerator.unwrap_model`] method to unwrap the model first before loading the weights. All model parameters are references to tensors, so this loads your weights inside `model`.

```py
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

</hfoption>
<hfoption id="sharded checkpoint">

Set `safe_serialization=True` to save the model in the safetensor format.

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

To load a sharded checkpoint or a safetensor formatted checkpoint, use the [`~accelerate.load_checkpoint_in_model`] method. This method allows you to load a checkpoint onto a specific device.

```py
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

</hfoption>
</hfoptions>

### State

During training, you may want to save the current state of the model, optimizer, random generators, and potentially learning rate schedulers so they can be restored in the *same script*. You should add the [`~Accelerator.save_state`] and [`~Accelerator.load_state`] methods to your script to save and load states.

To further customize where and how states are saved through [`~Accelerator.save_state`], use the [`~utils.ProjectConfiguration`] class. For example, if `automatic_checkpoint_naming` is enabled, each saved checkpoint is stored at `Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`.

Any other stateful items to be stored should be registered with the [`~Accelerator.register_for_checkpointing`] method so they can be saved and loaded. Every object passed to this method to be stored must have a `load_state_dict` and `state_dict` function.
