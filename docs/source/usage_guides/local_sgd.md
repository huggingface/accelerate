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

# Using Local SGD with 🤗 Accelerate

Local SGD is a technique for distributed training where gradients are not synchronized every step. Thus, each process updates its own version of the model weights and after a given number of steps these weights are synchronized by averaging across all processes. This improves communication efficiency and can lead to substantial training speed up especially when a computer lacks a faster interconnect such as NVLink.
Unlike gradient accumulation (where improving communication efficiency requires increasing the effective batch size), Local SGD does not require changing a batch size or a learning rate / schedule. However, if necessary, Local SGD can be combined with gradient accumulation as well.

In this tutorial you will see how to quickly setup  Local SGD 🤗 Accelerate. Compared to a standard Accelerate setup, this requires only two extra lines of code.

This example will use a very simplistic PyTorch training loop that performs gradient accumulation every two batches:

```python
device = "cuda"
model.to(device)

gradient_accumulation_steps = 2

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (index + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## Converting it to 🤗 Accelerate

First the code shown earlier will be converted to use 🤗 Accelerate  with neither a LocalSGD or a gradient accumulation helper:

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for index, batch in enumerate(training_dataloader):
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      loss = loss / gradient_accumulation_steps
+     accelerator.backward(loss)
      if (index+1) % gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()
```

## Letting 🤗 Accelerate handle model synchronization 

All that is left now is to let 🤗 Accelerate handle model parameter synchronization **and** the gradient accumulation for us. For simplicity let us assume we need to synchronize every 8 steps. This is
achieved by adding one `with LocalSGD` statement and one call `local_sgd.step()` after every optimizer step:

```diff
+local_sgd_steps=8

+with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
    for batch in training_dataloader:
        with accelerator.accumulate(model):
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
+           local_sgd.step()
```

Under the hood, the Local SGD code **disables** automatic gradient synchornization (but accumulation still works as expected!). Instead it averages model parameters every `local_sgd_steps` steps (as well as in the end of the training loop).

## Limitations

The current implementation works only with basic multi-GPU (or multi-CPU) training without, e.g., [DeepSpeed.](https://github.com/microsoft/DeepSpeed).

## References

    Although we are not aware of the true origins of this simple approach, the idea of local SGD is quite old and goes
    back to at least:

    Zhang, J., De Sa, C., Mitliagkas, I., & Ré, C. (2016). [Parallel SGD: When does averaging help?. arXiv preprint
    arXiv:1606.07365.](https://arxiv.org/abs/1606.07365)

    We credit the term Local SGD to the following paper (but there might be earlier references we are not aware of).

    Stich, Sebastian Urban. ["Local SGD Converges Fast and Communicates Little." ICLR 2019-International Conference on
    Learning Representations. No. CONF. 2019.](https://arxiv.org/abs/1805.09767)
