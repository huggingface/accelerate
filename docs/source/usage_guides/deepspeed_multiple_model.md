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

    This guide assumes that you have read and understood the [DeepSpeed usage guide](./deepspeed.md).

</Tip>

Running multiple models with Accelerate and DeepSpeed is useful for:

* Knowledge distillation
* Post-training techniques like RLHF (see the [TRL](https://github.com/huggingface/trl) library for more examples)
* Training multiple models at once

Currently, Accelerate has a **very experimental API** to help you use multiple models.

This tutorial will focus on two common use cases:

1. Knowledge distillation, where a smaller student model is trained to mimic a larger, better-performing teacher.  If the student model fits on a single GPU, we can use ZeRO-2 for training and ZeRO-3 to shard the teacher for inference. This is significantly faster than using ZeRO-3 for both models.
2. Training multiple *disjoint* models at once.

## Knowledge distillation

Knowledge distillation is a good example of using multiple models, but only training one of them.

Normally, you would use a single [`utils.DeepSpeedPlugin`] for both models. However, in this case, there are two separate configurations. Accelerate allows you to create and use multiple plugins **if and only if** they are in a `dict` so that you can reference and enable the proper plugin when needed.

```python
from accelerate.utils import DeepSpeedPlugin

zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")

deepspeed_plugins = {"student": zero2_plugin, "teacher": zero3_plugin}
```

The `zero2_config.json` should be configured for full training (so specify `scheduler` and `optimizer` if you are not utilizing your own), while `zero3_config.json` should only be configured for the inference model, as shown in the example below.

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

An example `zero2_config.json` configuration is shown below.

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

    DeepSpeed will raise an error if `train_micro_batch_size_per_gpu` isn't specified, even if this particular model isn't being trained.

</Tip>

From here, create a single [`Accelerator`] and pass in both configurations.

```python
from accelerate import Accelerator

accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
```

Now let's see how to use them.

### Student model

By default, Accelerate sets the first item in the `dict` as the default or enabled plugin (`"student"` plugin). Verify this by using the [`utils.deepspeed.get_active_deepspeed_plugin`] function to see which plugin is enabled.

```python
active_plugin = get_active_deepspeed_plugin(accelerator.state)
assert active_plugin is deepspeed_plugins["student"]
```

[`AcceleratorState`] also keeps the active DeepSpeed plugin saved in `state.deepspeed_plugin`.
```python
assert active_plugin is accelerator.deepspeed_plugin
```

Since `student` is the currently active plugin, let's go ahead and prepare the model, optimizer, and scheduler.

```python
student_model, optimizer, scheduler = ...
student_model, optimizer, scheduler, train_dataloader = accelerator.prepare(student_model, optimizer, scheduler, train_dataloader)
```

Now it's time to deal with the teacher model.

### Teacher model

First, you need to specify in [`Accelerator`] that the `zero3_config.json` configuration should be used.

```python
accelerator.state.select_deepspeed_plugin("teacher")
```

This disables the `"student"` plugin and enables the `"teacher"` plugin instead. The
DeepSpeed stateful config inside of Transformers is updated, and it changes which plugin configuration gets called when using
`deepspeed.initialize()`. This allows you to use the automatic `deepspeed.zero.Init`  context manager integration Transformers provides.

```python
teacher_model = AutoModel.from_pretrained(...)
teacher_model = accelerator.prepare(teacher_model)
```

Otherwise, you should manually initialize the model with `deepspeed.zero.Init`.
```python
with deepspeed.zero.Init(accelerator.deepspeed_plugin.config):
    model = MyModel(...)
```

### Training

From here, your training loop can be whatever you like, as long as `teacher_model` is never being trained on.

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

## Train multiple disjoint models

Training multiple models is a more complicated scenario.
In its current state, we assume each model is **completely disjointed** from the other during training.

This scenario still requires two [`utils.DeepSpeedPlugin`]'s to be made. However, you also need a second [`Accelerator`], since different `deepspeed` engines are being called at different times. A single [`Accelerator`] can only carry one instance at a time.

Since the [`state.AcceleratorState`] is a stateful object though, it is already aware of both [`utils.DeepSpeedPlugin`]'s available. You can just instantiate a second [`Accelerator`] with no extra arguments.

```python
first_accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
second_accelerator = Accelerator()
```

You can call either `first_accelerator.state.select_deepspeed_plugin()` to enable or disable
a particular plugin, and then call [`prepare`].

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

## Resources

To see more examples, please check out the [related tests](https://github.com/huggingface/accelerate/blob/main/src/accelerate/test_utils/scripts/external_deps/test_ds_multiple_model.py) currently in [Accelerate].
