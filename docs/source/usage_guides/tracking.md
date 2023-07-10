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

# Tracking

There are a large number of experiment tracking API's available, however getting them all to work with in a multi-processing environment can oftentimes be complex.
🤗 Accelerate provides a general tracking API that can be used to log useful items during your script through [`Accelerator.log`]

## Integrated Trackers

Currently `Accelerate` supports four trackers out-of-the-box:

- TensorBoard
- WandB
- CometML
- MLFlow

To use any of them, pass in the selected type(s) to the `log_with` parameter in [`Accelerate`]:
```python
from accelerate import Accelerator
from accelerate.utils import LoggerType

accelerator = Accelerator(log_with="all")  # For all available trackers in the environment
accelerator = Accelerator(log_with="wandb")
accelerator = Accelerator(log_with=["wandb", LoggerType.TENSORBOARD])
```

At the start of your experiment [`Accelerator.init_trackers`] should be used to setup your project, and potentially add any experiment hyperparameters to be logged:
```python
hps = {"num_iterations": 5, "learning_rate": 1e-2}
accelerator.init_trackers("my_project", config=hps)
```

When you are ready to log any data, [`Accelerator.log`] should be used.
A `step` can also be passed in to correlate the data with a particular step in the training loop.
```python
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=1)
```

Once you've finished training, make sure to run [`Accelerator.end_training`] so that all the trackers can run their finish functionalities if they have any.
```python
accelerator.end_training()
```


A full example is below:
```python
from accelerate import Accelerator

accelerator = Accelerator(log_with="all")
config = {
    "num_iterations": 5,
    "learning_rate": 1e-2,
    "loss_function": str(my_loss_function),
}

accelerator.init_trackers("example_project", config=config)

my_model, my_optimizer, my_training_dataloader = accelerate.prepare(my_model, my_optimizer, my_training_dataloader)
device = accelerator.device
my_model.to(device)

for iteration in config["num_iterations"]:
    for step, batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        accelerator.backward(loss)
        my_optimizer.step()
        accelerator.log({"training_loss": loss}, step=step)
accelerator.end_training()
```

If a tracker requires a directory to save data to, such as `TensorBoard`, then pass the directory path to `project_dir`. The `project_dir` parameter is useful 
when there are other configurations to be combined with in the [`~utils.ProjectConfiguration`] data class. For example, you can save the TensorBoard data to `project_dir` and everything else can be logged in the `logging_dir` parameter of [`~utils.ProjectConfiguration`: 

```python
accelerator = Accelerator(log_with="tensorboard", project_dir=".")

# use with ProjectConfiguration
config = ProjectConfiguration(project_dir=".", logging_dir="another/directory")
accelerator = Accelerator(log_with="tensorboard", project_config=config)
```

## Implementing Custom Trackers

To implement a new tracker to be used in `Accelerator`, a new one can be made through implementing the [`GeneralTracker`] class.
Every tracker must implement three functions and have three properties:
  - `__init__`: 
    - Should store a `run_name` and initialize the tracker API of the integrated library. 
    - If a tracker stores their data locally (such as TensorBoard), a `logging_dir` parameter can be added.
  - `store_init_configuration`: 
    - Should take in a `values` dictionary and store them as a one-time experiment configuration
  - `log`: 
    - Should take in a `values` dictionary and a `step`, and should log them to the run

  - `name` (`str`):
    - A unique string name for the tracker, such as `"wandb"` for the wandb tracker. 
    - This will be used for interacting with this tracker specifically
  - `requires_logging_directory` (`bool`):
    - Whether a `logging_dir` is needed for this particular tracker and if it uses one.
  - `tracker`: 
    - This should be implemented as a `@property` function 
    - Should return the internal tracking mechanism the library uses, such as the `run` object for `wandb`.

Each method should also utilize the [`state.PartialState`] class if the logger should only be executed on the main process for instance.

A brief example can be seen below with an integration with Weights and Biases, containing only the relevant information and logging just on 
the main process:
```python
from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb


class MyCustomTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str):
        self.run_name = run_name
        run = wandb.init(self.run_name)

    @property
    def tracker(self):
        return self.run.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)
```

When you are ready to build your `Accelerator` object, pass in an **instance** of your tracker to [`Accelerator.log_with`] to have it automatically
be used with the API:

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=tracker)
```

These also can be mixed with existing trackers, including with `"all"`:

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=[tracker, "all"])
```

## Accessing the internal tracker 

If some custom interactions with a tracker might be wanted directly, you can quickly access one using the 
[`Accelerator.get_tracker`] method. Just pass in the string corresponding to a tracker's `.name` attribute 
and it will return that tracker on the main process.

This example shows doing so with wandb:

```python
wandb_tracker = accelerator.get_tracker("wandb")
```

From there you can interact with `wandb`'s `run` object like normal:

```python
wandb_run.log_artifact(some_artifact_to_log)
```

<Tip>
  Trackers built in Accelerate will automatically execute on the correct process, 
  so if a tracker is only meant to be ran on the main process it will do so 
  automatically.
</Tip>

If you want to truly remove Accelerate's wrapping entirely, you can 
achieve the same outcome with:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```


## When a wrapper cannot work

If a library has an API that does not follow a strict `.log` with an overall dictionary such as Neptune.AI, logging can be done manually under an `if accelerator.is_main_process` statement:
```diff
  from accelerate import Accelerator
+ import neptune.new as neptune

  accelerator = Accelerator()
+ run = neptune.init(...)

  my_model, my_optimizer, my_training_dataloader = accelerate.prepare(my_model, my_optimizer, my_training_dataloader)
  device = accelerator.device
  my_model.to(device)

  for iteration in config["num_iterations"]:
      for batch in my_training_dataloader:
          my_optimizer.zero_grad()
          inputs, targets = batch
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = my_model(inputs)
          loss = my_loss_function(outputs, targets)
          total_loss += loss
          accelerator.backward(loss)
          my_optimizer.step()
+         if accelerator.is_main_process:
+             run["logs/training/batch/loss"].log(loss)
```
