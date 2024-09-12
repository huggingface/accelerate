<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Using multiple models with DeepSpeed

<Tip warning={true}>

    This guide assumes that you have read and understand the [DeepSpeed usage guide](./deepspeed.md).

</Tip>

## Use cases

There are a few use cases where one may want to run multiple models when using `accelerate` and `DeepSpeed`:

* Knowledge distillation
* RLHF techniques (see the `trl` repo for [more examples](https://github.com/huggingface/trl))
* Training multiple models at once.

As it stands, Accelerate has a **very experimental API** to help you accomplish this.

This tutorial will focus on two common use cases:

1. Using `zero2` to train a model, and `zero3` to perform inference for assisting of the training for the `zero2` model (as `zero2` is faster to train on if the model fits onto a single GPU), which is shown in the RLHF example
2. Training multiple *disjoint* models at once, shown later

## Multiple models, only one is trained

RLHF is a good example of such use-case

Normally, you would use a single [`utils.DeepSpeedPlugin`] at a time, however in this case we have two separate configurations. Accelerate allows you to create and use multiple plugins **if and only if** they are in a `dict` so that we can reference/enable the proper plugin when needed:

```python
from accelerate.utils import DeepSpeedPlugin

zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")

deepspeed_plugins = {"student": zero2_plugin, "teacher": zero3_plugin}
```

When doing so,  `zero2_config.json` should be configured for full training (so specifying `scheduler` and `optimizer` if not utilizing your own) while `zero3_config.json` should *only* be configured for the inference model,
such as the following example for a `zero3_config.json`:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
    },
    "train_micro_batch_size_per_gpu": 1
}
```

And the zero2 counterpart:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": true,
            "adam_w_mode": true
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
```

<Tip>

    DeepSpeed will raise an error if we don't specify a `train_micro_batch_size_per_gpu`, despite not really 
    training this particular model

</Tip>

From here, we can create a single [`Accelerator`] and pass in both configurations:

```python
from accelerate import Accelerator

accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
```

Now let's see how to use them.

### The student model

By default, `accelerate` will set the first item in the `dict` as the default/enabled plugin (so here our `"student"` plugin). We can verify this by using the [`utils.deepspeed.get_active_deepspeed_plugin]` function to see which one is being used:

```python
active_plugin = get_active_deepspeed_plugin(accelerator.state)
assert active_plugin is deepspeed_plugins["student"]
```

Also the `AcceleratorState` will keep the active DeepSpeed plugin saved in `state.deepspeed_plugin`:
```python
assert active_plugin is accelerator.deepspeed_plugin
```

Since the `student` is the currently active plugin, let's go ahead and prepare our model, optimizer, and scheduler like normal:

```python
student_model, optimizer, scheduler = ...
student_model, optimizer, scheduler, train_dataloader = accelerator.prepare(student_model, optimizer, scheduler, train_dataloader)
```

From here, we need to deal with the teacher model.

### The teacher model

First we need to tell the `Accelerator` that the zero3/teacher ZeRO configuration should be used:

```python
accelerator.state.select_deepspeed_plugin("teacher")
```

Doing so will disable the `"student"` plugin and enable the `"teacher"` one instead. This will update the
DeepSpeed stateful config inside of `transformers`, and change which plugin configuration gets called when using
`deepspeed.initialize()`, allowing us to use the no-code `deepspeed.zero.Init`  context manager `transformers` provides:

```python
teacher_model = AutoModel.from_pretrained(...)
teacher_model = accelerator.prepare(teacher_model)
```

Otherwise you should manually initialize them under `deepspeed.zero.Init`:
```python
with deepspeed.zero.Init(accelerator.deepspeed_plugin.config):
    model = MyModel(...)
```

### Training

From here, your training loop can be whatever you like, making sure that `teacher_model` is never being trained on:

```python
teacher_model.eval()
student_model.train()
for batch in train_dataloader:
    with torch.no_grad():
        output_teacher = teacher_model(**batch)
    output_student = student_model(**batch)
    # Combine the losses or modify it in some way
    loss = output_teacher.loss + output_student.loss
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## Multiple models all being trained

Multiple models being trained is a more complicated scenario and is actively being looked upon by the team.
In its current state, we assume that each model is **completely disjoint** from the other during training.

With this situation, we still require two [`utils.DeepSpeedPlugin`]'s to be made, however we also further
require a second `Accelerator`, since we need to call different `deepspeed` engines at different times, and a single `Accelerator` only
carries one instance of it at a time.

Since the [`state.AcceleratorState`] is a stateful object though, it is already aware of both `DeepSpeedPlugin`'s available,
meaning you can just instantiate a second `Accelerator` with no extra arguments:

```python
first_accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
second_accelerator = Accelerator()
```

Similar to before, we can call either `first_accelerator.state.select_deepspeed_plugin()` to enable/disable
a particular plugin, and call `.prepare` like normal:

```python
# can be `accelerator_0`, `accelerator_1`, or by calling `AcceleratorState().select_deepspeed_plugin(...)`
first_accelerator.state.select_deepspeed_plugin("first_model")
first_model = AutoModel.from_pretrained(...)
# For this example, `get_training_items` is a nonexistent function that gets the setup we need for training
first_optimizer, first_scheduler, train_dl, eval_dl = get_training_items(model1)
first_model, first_optimizer, first_scheduler, train_dl, eval_dl = accelerator.prepare(
    first_model, first_optimizer, first_scheduler, train_dl, eval_dl
)

second_accelerator.state.select_deepspeed_plugin("second_model")
second_model = AutoModel.from_pretrained(...)
# For this example, `get_training_items` is a nonexistent function that gets the setup we need for training
second_optimizer, second_scheduler, _, _ = get_training_items(model2)
second_model, second_optimizer, second_scheduler = accelerator.prepare(
    second_model, second_optimizer, second_scheduler
)
```

And now you can train:

```python
for batch in dl:
    outputs1 = first_model(**batch)
    first_accelerator.backward(outputs1.loss)
    first_optimizer.step()
    first_scheduler.step()
    first_optimizer.zero_grad()
    
    outputs2 = model2(**batch)
    second_accelerator.backward(outputs2.loss)
    second_optimizer.step()
    second_scheduler.step()
    second_optimizer.zero_grad()
```

## More Resources

To see more examples, please check out the [related tests](https://github.com/huggingface/accelerate/blob/main/src/accelerate/test_utils/scripts/external_deps/test_ds_multiple_model.py) we currently have in `Accelerate`, and we will be improving this further
with better examples, documentation, and functionality as we continue to flesh out this API
