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

# Training on TPUs with 🤗 Accelerate

Training on TPUs can be slightly different from training on multi-gpu, even with 🤗 Accelerate. This guide aims to show you 
where you should be careful and why, as well as the best practices in general.

## Training in a Notebook

The main carepoint when training on TPUs comes from the [`notebook_launcher`]. As mentioned in the [notebook tutorial](../usage_guides/notebook), you need to 
restructure your training code into a function that can get passed to the [`notebook_launcher`] function and be careful about not declaring any tensors on the GPU.

While on a TPU that last part is not as important, a critical part to understand is that when you launch code from a notebook you do so through a process called **forking**. 
When launching from the command-line, you perform **spawning**, where a python process is not currently running and you *spawn* a new process in. Since your Jupyter notebook is already 
utilizing a python process, you need to *fork* a new process from it to launch your code. 

Where this becomes important is in regard to declaring your model. On forked TPU processes, it is recommended that you instantiate your model *once* and pass this into your 
training function. This is different than training on GPUs where you create `n` models that have their gradients synced and back-propagated at certain moments. Instead, one 
model instance is shared between all the nodes and it is passed back and forth. This is important especially when training on low-resource TPUs such as those provided in Kaggle kernels or
on Google Colaboratory. 

Below is an example of a training function passed to the [`notebook_launcher`] if training on CPUs or GPUs:

<Tip>

    This code snippet is based off the one from the `simple_nlp_example` notebook found [here](https://github.com/huggingface/notebooks/blob/main/examples/accelerate/simple_nlp_example.ipynb) with slight 
    modifications for the sake of simplicity

</Tip>

```python
def training_function():
    # Initialize accelerator
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
```

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

<Tip>

    The `notebook_launcher` will default to 8 processes if 🤗 Accelerate has been configured for a TPU

</Tip>

If you use this example and declare the model *inside* the training loop, then on a low-resource system you will potentially see an error 
like:

```
ProcessExitedException: process 0 terminated with signal SIGSEGV
```

This error is *extremely* cryptic but the basic explanation is you ran out of system RAM. You can avoid this entirely by reconfiguring the training function to 
accept a single `model` argument, and declare it in an outside cell:

```python
# In another Jupyter cell
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

```diff
+ def training_function(model):
      # Initialize accelerator
      accelerator = Accelerator()
-     model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
      train_dataloader, eval_dataloader = create_dataloaders(
          train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
      )
  ...
```

And finally calling the training function with:

```diff
  from accelerate import notebook_launcher
- notebook_launcher(training_function)
+ notebook_launcher(training_function, (model,))
```

<Tip>

    The above workaround is only needed when launching a TPU instance from a Jupyter Notebook on a low-resource server such as Google Colaboratory or Kaggle. If 
    using a script or launching on a much beefier server declaring the model beforehand is not needed.

</Tip>

## Mixed Precision and Global Variables 

As mentioned in the [mixed precision tutorial](../usage_guides/mixed_precision), 🤗 Accelerate supports fp16 and bf16, both of which can be used on TPUs.
That being said, ideally `bf16` should be utilized as it is extremely efficient to use.

There are two "layers" when using `bf16` and 🤗 Accelerate on TPUs, at the base level and at the operation level. 

At the base level, this is enabled when passing `mixed_precision="bf16"` to `Accelerator`, such as:
```python
accelerator = Accelerator(mixed_precision="bf16")
```
By default, this will cast `torch.float` and `torch.double` to `bfloat16` on TPUs. 
The specific configuration being set is an environmental variable of `XLA_USE_BF16` is set to `1`.

There is a further configuration you can perform which is setting the `XLA_DOWNCAST_BF16` environmental variable. If set to `1`, then 
`torch.float` is `bfloat16` and `torch.double` is `float32`.

This is performed in the `Accelerator` object when passing `downcast_bf16=True`:
```python
accelerator = Accelerator(mixed_precision="bf16", downcast_bf16=True)
```

Using downcasting instead of bf16 everywhere is good for when you are trying to calculate metrics, log values, and more where raw bf16 tensors would be unusable. 

## Training Times on TPUs

As you launch your script, you may notice that training seems exceptionally slow at first. This is because TPUs
first run through a few batches of data to see how much memory to allocate before finally utilizing this configured 
memory allocation extremely efficiently. 

If you notice that your evaluation code to calculate the metrics of your model takes longer due to a larger batch size being used, 
it is recommended to keep the batch size the same as the training data if it is too slow. Otherwise the memory will reallocate to this 
new batch size after the first few iterations. 

<Tip>

    Just because the memory is allocated does not mean it will be used or that the batch size will increase when going back to your training dataloader.

</Tip>
