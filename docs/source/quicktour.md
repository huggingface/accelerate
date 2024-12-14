<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Quicktour

There are many ways to launch and run your code depending on your training environment ([torchrun](https://pytorch.org/docs/stable/elastic/run.html), [DeepSpeed](https://www.deepspeed.ai/), etc.) and available hardware. Accelerate offers a unified interface for launching and training on different distributed setups, allowing you to focus on your PyTorch training code instead of the intricacies of adapting your code to these different setups. This allows you to easily scale your PyTorch code for training and inference on distributed setups with hardware like GPUs and TPUs. Accelerate also provides Big Model Inference to make loading and running inference with really large models that usually don't fit in memory more accessible.

This quicktour introduces the three main features of Accelerate:

* a unified command line launching interface for distributed training scripts
* a training library for adapting PyTorch training code to run on different distributed setups
* Big Model Inference

## Unified launch interface

Accelerate automatically selects the appropriate configuration values for any given distributed training framework (DeepSpeed, FSDP, etc.) through a unified configuration file generated from the [`accelerate config`](package_reference/cli#accelerate-config) command. You could also pass the configuration values explicitly to the command line which is helpful in certain situations like if you're using SLURM.


But in most cases, you should always run [`accelerate config`](package_reference/cli#accelerate-config) first to help Accelerate learn about your training setup.

```bash
accelerate config
```

The [`accelerate config`](package_reference/cli#accelerate-config) command creates and saves a default_config.yaml file in Accelerates cache folder. This file stores the configuration for your training environment, which helps Accelerate correctly launch your training script based on your machine.

After you've configured your environment, you can test your setup with [`accelerate test`](package_reference/cli#accelerate-test), which launches a short script to test the distributed environment.

```bash
accelerate test
```

> [!TIP]
> Add `--config_file` to the `accelerate test` or `accelerate launch` command to specify the location of the configuration file if it is saved in a non-default location like the cache.

Once your environment is setup, launch your training script with [`accelerate launch`](package_reference/cli#accelerate-launch)!

```bash
accelerate launch path_to_script.py --args_for_the_script
```

To learn more, check out the [Launch distributed code](basic_tutorials/launch) tutorial for more information about launching your scripts.

We also have a [configuration zoo](https://github.com/huggingface/accelerate/blob/main/examples/config_yaml_templates) which showcases a number of premade **minimal** example configurations for a variety of setups you can run.

## Adapt training code

The next main feature of Accelerate is the [`Accelerator`] class which adapts your PyTorch code to run on different distributed setups.

You only need to add a few lines of code to your training script to enable it to run on multiple GPUs or TPUs.

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ device = accelerator.device
+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```

1. Import and instantiate the [`Accelerator`] class at the beginning of your training script. The [`Accelerator`] class initializes everything necessary for distributed training, and it automatically detects your training environment (a single machine with a GPU, a machine with several GPUs, several machines with multiple GPUs or a TPU, etc.) based on how the code was launched.

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

2. Remove calls like `.cuda()` on your model and input data. The [`Accelerator`] class automatically places these objects on the appropriate device for you.

> [!WARNING]
> This step is *optional* but it is considered best practice to allow Accelerate to handle device placement. You could also deactivate automatic device placement by passing `device_placement=False` when initializing the [`Accelerator`]. If you want to explicitly place objects on a device with `.to(device)`, make sure you use `accelerator.device` instead. For example, if you create an optimizer before placing a model on `accelerator.device`, training fails on a TPU.

> [!WARNING]
> Accelerate does not use non-blocking transfers by default for its automatic device placement, which can result in potentially unwanted CUDA synchronizations.  You can enable non-blocking transfers by passing a [`~utils.dataclasses.DataLoaderConfiguration`] with `non_blocking=True` set as the `dataloader_config` when initializing the [`Accelerator`].  As usual, non-blocking transfers will only work if the dataloader also has `pin_memory=True` set.  Be wary that using non-blocking transfers from GPU to CPU may cause incorrect results if it results in CPU operations being performed on non-ready tensors.

```py
device = accelerator.device
```

3. Pass all relevant PyTorch objects for training (optimizer, model, dataloader(s), learning rate scheduler) to the [`~Accelerator.prepare`] method as soon as they're created. This method wraps the model in a container optimized for your distributed setup, uses Accelerates version of the optimizer and scheduler, and creates a sharded version of your dataloader for distribution across GPUs or TPUs.

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

4. Replace `loss.backward()` with [`~Accelerator.backward`] to use the correct `backward()` method for your training setup.

```py
accelerator.backward(loss)
```

Read [Accelerate’s internal mechanisms](concept_guides/internal_mechanism) guide to learn more details about how Accelerate adapts your code.

### Distributed evaluation

To perform distributed evaluation, pass your validation dataloader to the [`~Accelerator.prepare`] method:

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```

Each device in your distributed setup only receives a part of the evaluation data, which means you should group your predictions together with the [`~Accelerator.gather_for_metrics`] method. This method requires all tensors to be the same size on each process, so if your tensors have different sizes on each process (for instance when dynamically padding to the maximum length in a batch), you should use the [`~Accelerator.pad_across_processes`] method to pad you tensor to the largest size across processes. Note that the tensors needs to be 1D and that we concatenate the tensors along the first dimension. 

```python
for inputs, targets in validation_dataloader:
    predictions = model(inputs)
    # Gather all predictions and targets
    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
    # Example of use with a *Datasets.Metric*
    metric.add_batch(all_predictions, all_targets)
```

For more complex cases (e.g. 2D tensors, don't want to concatenate tensors, dict of 3D tensors), you can pass `use_gather_object=True` in `gather_for_metrics`. This will return the list of objects after gathering. Note that using it with GPU tensors is not well supported and inefficient.

> [!TIP]
> Data at the end of a dataset may be duplicated so the batch can be equally divided among all workers. The [`~Accelerator.gather_for_metrics`] method automatically removes the duplicated data to calculate a more accurate metric.

## Big Model Inference

Accelerate's Big Model Inference has two main features, [`~accelerate.init_empty_weights`] and [`~accelerate.load_checkpoint_and_dispatch`], to load large models for inference that typically don't fit into memory.

> [!TIP]
> Take a look at the [Handling big models for inference](concept_guides/big_model_inference) guide for a better understanding of how Big Model Inference works under the hood.

### Empty weights initialization

The [`~accelerate.init_empty_weights`] context manager initializes models of any size by creating a *model skeleton* and moving and placing parameters each time they're created to PyTorch's [**meta**](https://pytorch.org/docs/main/meta.html) device. This way, not all weights are immediately loaded and only a small part of the model is loaded into memory at a time.

For example, loading an empty [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model takes significantly less memory than fully loading the models and weights on the CPU.

```py
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

### Load and dispatch weights

The [`~accelerate.load_checkpoint_and_dispatch`] function loads full or sharded checkpoints into the empty model, and automatically distribute weights across all available devices.

The `device_map` parameter determines where to place each model layer, and specifiying `"auto"` places them on the GPU first, then the CPU, and finally the hard drive as memory-mapped tensors if there's still not enough memory. Use the `no_split_module_classes` parameter to indicate which modules shouldn't be split across devices (typically those with a residual connection).

```py
from accelerate import load_checkpoint_and_dispatch

model_checkpoint = "your-local-model-folder"
model = load_checkpoint_and_dispatch(
    model, checkpoint=model_checkpoint, device_map="auto", no_split_module_classes=['Block']
)
```

## Next steps

Now that you've been introduced to the main Accelerate features, your next steps could include:

* Check out the [tutorials](basic_tutorials/overview) for a gentle walkthrough of Accelerate. This is especially useful if you're new to distributed training and the library.
* Dive into the [guides](usage_guides/explore) to see how to use Accelerate for specific use-cases.
* Deepen your conceptual understanding of how Accelerate works internally by reading the [concept guides](concept_guides/internal_mechanism).
* Look up classes and commands in the [API reference](package_reference/accelerator) to see what parameters and options are available.
