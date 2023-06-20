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

# Checkpointing

When training a PyTorch model with 🤗 Accelerate, you may often want to save and continue a state of training. Doing so requires
saving and loading the model, optimizer, RNG generators, and the GradScaler. Inside 🤗 Accelerate are two convenience functions to achieve this quickly:
- Use [`~Accelerator.save_state`] for saving everything mentioned above to a folder location
- Use [`~Accelerator.load_state`] for loading everything stored from an earlier `save_state`

To further customize where and how states are saved through [`~Accelerator.save_state`] the [`~utils.ProjectConfiguration`] class can be used. For example 
if `automatic_checkpoint_naming` is enabled each saved checkpoint will be located then at `Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`.

It should be noted that the expectation is that those states come from the same training script, they should not be from two separate scripts.

- By using [`~Accelerator.register_for_checkpointing`], you can register custom objects to be automatically stored or loaded from the two prior functions,
so long as the object has a `state_dict` **and** a `load_state_dict` functionality. This could include objects such as a learning rate scheduler. 


Below is a brief example using checkpointing to save and reload a state during training:

```python
from accelerate import Accelerator
import torch

accelerator = Accelerator(project_dir="my/save/path")

my_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size=1, gamma=0.99)
my_model, my_optimizer, my_training_dataloader = accelerator.prepare(my_model, my_optimizer, my_training_dataloader)

# Register the LR scheduler
accelerator.register_for_checkpointing(my_scheduler)

# Save the starting state
accelerator.save_state()

device = accelerator.device
my_model.to(device)

# Perform training
for epoch in range(num_epochs):
    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        accelerator.backward(loss)
        my_optimizer.step()
    my_scheduler.step()

# Restore the previous state
accelerator.load_state("my/save/path/checkpointing/checkpoint_0")
```

## Restoring the state of the DataLoader 

After resuming from a checkpoint, it may also be desirable to resume from a particular point in the active `DataLoader` if 
the state was saved during the middle of an epoch. You can use [`~Accelerator.skip_first_batches`] to do so. 

```python
from accelerate import Accelerator

accelerator = Accelerator(project_dir="my/save/path")

train_dataloader = accelerator.prepare(train_dataloader)
accelerator.load_state("my_state")

# Assume the checkpoint was saved 100 steps into the epoch
skipped_dataloader = accelerator.skip_first_batches(train_dataloader, 100)

# After the first iteration, go back to `train_dataloader`

# First epoch
for batch in skipped_dataloader:
    # Do something
    pass

# Second epoch
for batch in train_dataloader:
    # Do something
    pass
```
