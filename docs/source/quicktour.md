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

# Quick tour

Let's have a look at the 🤗 Accelerate main features and traps to avoid.

## Main use

To use 🤗 Accelerate in your own script, you have to change four things:

1. Import the [`Accelerator`] main class and instantiate one in an `accelerator` object:

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

This should happen as early as possible in your training script as it will initialize everything necessary for
distributed training. You don't need to indicate the kind of environment you are in (just one machine with a GPU, one
machines with several GPUs, several machines with multiple GPUs or a TPU), the library will detect this automatically.

2. Remove the call `.to(device)` or `.cuda()` for your model and input data. The `accelerator` object
will handle this for you and place all those objects on the right device for you. If you know what you're doing, you
can leave those `.to(device)` calls but you should use the device provided by the `accelerator` object:
`accelerator.device`.

To fully deactivate the automatic device placement, pass along `device_placement=False` when initializing your
[`Accelerator`].

<Tip warning={true}>

    If you place your objects manually on the proper device, be careful to create your optimizer after putting your
    model on `accelerator.device` or your training will fail on TPU.

</Tip>

3. Pass all objects relevant to training (optimizer, model, training dataloader, learning rate scheduler) to the
[`~Accelerator.prepare`] method. This will make sure everything is ready for training.

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

In particular, your training dataloader will be sharded across all GPUs/TPU cores available so that each one sees a
different portion of the training dataset. Also, the random states of all processes will be synchronized at the
beginning of each iteration through your dataloader, to make sure the data is shuffled the same way (if you decided to
use `shuffle=True` or any kind of random sampler).

<Tip>

    The actual batch size for your training will be the number of devices used multiplied by the batch size you set in
    your script: for instance training on 4 GPUs with a batch size of 16 set when creating the training dataloader will
    train at an actual batch size of 64.

</Tip>

Alternatively, you can use the option `split_batches=True` when creating and initializing your
[`Accelerator`], in which case the batch size will always stay the same, whether you run your
script on 1, 2, 4, or 64 GPUs.

You should execute this instruction as soon as all objects for training are created, before starting your actual
training loop.

<Tip warning={true}>

    You should only pass the learning rate scheduler to [`~Accelerator.prepare`] when the scheduler needs to be stepped
    at each optimizer step.

</Tip>

<Tip warning={true}>

    Your training dataloader may change length when going through this method: if you run on X GPUs, it will have its
    length divided by X (since your actual batch size will be multiplied by X), unless you set
    `split_batches=True`.

</Tip>

Any instruction using your training dataloader length (for instance if you want to log the number of total training
steps) should go after the call to [`~Accelerator.prepare`].

You can perfectly send your dataloader to [`~Accelerator.prepare`] on its own, but it's best to send the
model and optimizer to [`~Accelerator.prepare`] together.

You may or may not want to send your validation dataloader to [`~Accelerator.prepare`], depending on
whether you want to run distributed evaluation or not (see below).

4. Replace the line `loss.backward()` by `accelerator.backward(loss)`.

And you're all set! With all these changes, your script will run on your local machine as well as on multiple GPUs or a
TPU! You can either use your favorite tool to launch the distributed training, or you can use the 🤗 Accelerate
launcher.


## Distributed evaluation

You can perform regular evaluation in your training script, if you leave your validation dataloader out of the
[`~Accelerator.prepare`] method. In this case, you will need to put the input data on the
`accelerator.device` manually.

To perform distributed evaluation, send along your validation dataloader to the [`~Accelerator.prepare`]
method:

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```

As for your training dataloader, it will mean that (should you run your script on multiple devices) each device will
only see part of the evaluation data. This means you will need to group your predictions together. This is very easy to
do with the [`~Accelerator.gather_for_metrics`] method.

```python
for inputs, targets in validation_dataloader:
    predictions = model(inputs)
    # Gather all predictions and targets
    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
    # Example of use with a *Datasets.Metric*
    metric.add_batch(all_predictions, all_targets)
```

<Tip warning={true}>

    Similar to the training dataloader, passing your validation dataloader through
    [`~Accelerator.prepare`] may change it: if you run on X GPUs, it will have its length divided by X
    (since your actual batch size will be multiplied by X), unless you set `split_batches=True`.

</Tip>

Any instruction using your training dataloader length (for instance if you need the number of total training steps
to create a learning rate scheduler) should go after the call to [`~Accelerator.prepare`]. 

Some data at the end of the dataset may be duplicated so the batch can be divided equally among all workers. As a result, metrics
should be calculated through the [`~Accelerator.gather_for_metrics`] method to automatically remove the duplicated data while gathering.

<Tip>

    If for some reason you don't wish to have this automatically done, [`~Accelerator.gather`] can be used instead to gather 
    the data across all processes and this can manually be done instead.

</Tip>


<Tip warning={true}>

    The [`~Accelerator.gather`] and [`~Accelerator.gather_for_metrics`] methods require the tensors to be all the same size on each process. If
    you have tensors of different sizes on each process (for instance when dynamically padding to the maximum length in
    a batch), you should use the [`~Accelerator.pad_across_processes`] method to pad you tensor to the
    biggest size across processes.

</Tip>

## Launching your distributed script

You can use the regular commands to launch your distributed training (like `torch.distributed.run` for
PyTorch), they are fully compatible with 🤗 Accelerate.

🤗 Accelerate also provides a CLI tool that unifies all launchers, so you only have to remember one command. To use it,
just run:

```bash
accelerate config
```

on your machine and reply to the questions asked. This will save a *default_config.yaml* file in your cache folder for
🤗 Accelerate. That cache folder is (with decreasing order of priority):

- The content of your environment variable `HF_HOME` suffixed with *accelerate*.
- If it does not exist, the content of your environment variable `XDG_CACHE_HOME` suffixed with
  *huggingface/accelerate*.
- If this does not exist either, the folder *~/.cache/huggingface/accelerate*

You can also specify with the flag `--config_file` the location of the file you want to save.

Once this is done, you can test everything is going well on your setup by running:

```bash
accelerate test
```

This will launch a short script that will test the distributed environment. If it runs fine, you are ready for the next
step!

Note that if you specified a location for the config file in the previous step, you need to pass it here as well:

```bash
accelerate test --config_file path_to_config.yaml
```

Now that this is done, you can run your script with the following command:

```bash
accelerate launch path_to_script.py --args_for_the_script
```

If you stored the config file in a non-default location, you can indicate it to the launcher like this:

```bash
accelerate launch --config_file path_to_config.yaml path_to_script.py --args_for_the_script
```

You can also override any of the arguments determined by your config file. 
To see the complete list of parameters that you can pass in, run `accelerate launch -h`. 

Check out the [Launch tutorial](basic_tutorials/launch) for more information about launching your scripts. 


## Launching training from a notebook

In Accelerate 0.3.0, a new [`notebook_launcher`] has been introduced to help you launch your training
function from a notebook. This launcher supports launching a training with TPUs on Colab or Kaggle, as well as training
on several GPUs (if the machine on which you are running your notebook has them).

Just define a function responsible for your whole training and/or evaluation in a cell of the notebook, then execute a
cell with the following code:

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

<Tip warning={true}>

    Your [`Accelerator`] object should only be defined inside the training function. This is because the
    initialization should be done inside the launcher only.

</Tip>

Check out the [Notebook Launcher tutorial](basic_tutorials/notebook) for more information about training on TPUs. 


## Training on TPU

If you want to launch your script on TPUs, there are a few caveats you should be aware of. Behind the scenes, the TPUs
will create a graph of all the operations happening in your training step (forward pass, backward pass and optimizer
step). This is why your first step of training will always be very long as building and compiling this graph for
optimizations takes some time.

The good news is that this compilation will be cached so the second step and all the following will be much faster. The
bad news is that it only applies if all of your steps do exactly the same operations, which implies:

- having all tensors of the same length in all your batches
- having static code (i.e., not a for loop of length that could change from step to step)

Having any of the things above change between two steps will trigger a new compilation which will, once again, take a
lot of time. In practice, that means you must take special care to have all your tensors in your inputs of the same
shape (so no dynamic padding for instance if you are in an NLP problem) and should not use layers with for loops that
have different lengths depending on the inputs (such as an LSTM) or the training will be excruciatingly slow.

To introduce special behavior in your script for TPUs you can check the `distributed_type` of your
`accelerator`:

```python docstyle-ignore
from accelerate import DistributedType

if accelerator.distributed_type == DistributedType.TPU:
    # do something of static shape
else:
    # go crazy and be dynamic
```

The [NLP example](https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py) shows an example in a 
situation with dynamic padding.

One last thing to pay close attention to: if your model has tied weights (such as language models which tie the weights
of the embedding matrix with the weights of the decoder), moving this model to the TPU (either yourself or after you
passed your model to [`~Accelerator.prepare`]) will break the tying. You will need to retie the weights
after. You can find an example of this in the [run_clm_no_trainer](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py) script in
the Transformers repository.

Check out the [TPU tutorial](concept_guides/training_tpu) for more information about training on TPUs. 


## Other caveats

We list here all smaller issues you could have in your script conversion and how to resolve them.

### Execute a statement only on one processes

Some of your instructions only need to run for one process on a given server: for instance a data download or a log
statement. To do this, wrap the statement in a test like this:

```python docstyle-ignore
if accelerator.is_local_main_process:
    # Is executed once per server
```

Another example is progress bars: to avoid having multiple progress bars in your output, you should only display one on
the local main process:

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
```

The *local* means per machine: if you are running your training on two servers with several GPUs, the instruction will
be executed once on each of those servers. If you need to execute something only once for all processes (and not per
machine) for instance, uploading the final model to the 🤗 model hub, wrap it in a test like this:

```python docstyle-ignore
if accelerator.is_main_process:
    # Is executed once only
```

For printing statements you only want executed once per machine, you can just replace the `print` function by
`accelerator.print`.


### Defer execution

When you run your usual script, instructions are executed in order. Using 🤗 Accelerate to deploy your script on several
GPUs at the same time introduces a complication: while each process executes all instructions in order, some may be
faster than others.

You might need to wait for all processes to have reached a certain point before executing a given instruction. For
instance, you shouldn't save a model before being sure every process is done with training. To do this, just write the
following line in your code:

```
accelerator.wait_for_everyone()
```

This instruction will block all the processes that arrive first until all the other processes have reached that
point (if you run your script on just one GPU or CPU, this won't do anything).


### Saving/loading a model

Saving the model you trained might need a bit of adjustment: first you should wait for all processes to reach that
point in the script as shown above, and then, you should unwrap your model before saving it. This is because when going
through the [`~Accelerator.prepare`] method, your model may have been placed inside a bigger model,
which deals with the distributed training. This in turn means that saving your model state dictionary without taking
any precaution will take that potential extra layer into account, and you will end up with weights you can't load back
in your base model. The [`~Accelerator.save_model`] method will help you to achieve that. It will unwrap your model and save
the model state dictionnary.

Here is an example:
```
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```
The [`~Accelerator.save_model`] method can also save a model into sharded checkpoints or with safetensors format.
Here is an example: 

```python
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

If your script contains logic to load a checkpoint, we also recommend you load your weights in the unwrapped model
(this is only useful if you use the load function after making your model go through
[`~Accelerator.prepare`]). Here is an example:

```python
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

Note that since all the model parameters are references to tensors, this will load your weights inside `model`.

If you want to load a sharded checkpoint or a checkpoint with safetensors format into the model with a specific `device`, we recommend you to load it with [`~utils.load_checkpoint_in_model`] function. Here's an example:

```python
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

## Saving/loading entire states

When training your model, you may want to save the current state of the model, optimizer, random generators, and potentially LR schedulers to be restored in the _same script_.
You can use [`~Accelerator.save_state`] and [`~Accelerator.load_state`] respectively to do so.

To further customize where and how states saved through [`~Accelerator.save_state`] the [`~utils.ProjectConfiguration`] class can be used. For example 
if `automatic_checkpoint_naming` is enabled each saved checkpoint will be located then at `Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`.

If you have registered any other stateful items to be stored through [`~Accelerator.register_for_checkpointing`] they will also be saved and/or loaded.

<Tip>

    Every object passed to [`~Accelerator.register_for_checkpointing`] must have a `load_state_dict` and `state_dict` function to be stored

</Tip>


### Gradient clipping

If you are using gradient clipping in your script, you should replace the calls to
`torch.nn.utils.clip_grad_norm_` or `torch.nn.utils.clip_grad_value_` with [`~Accelerator.clip_grad_norm_`]
and [`~Accelerator.clip_grad_value_`] respectively.


### Mixed Precision training

If you are running your training in Mixed Precision with 🤗 Accelerate, you will get the best result with your loss being
computed inside your model (like in Transformer models for instance). Every computation outside of the model will be
executed in full precision (which is generally what you want for loss computation, especially if it involves a
softmax). However you might want to put your loss computation inside the [`~Accelerator.autocast`] context manager:

```
with accelerator.autocast():
    loss = complex_loss_function(outputs, target):
```

Another caveat with Mixed Precision training is that the gradient will skip a few updates at the beginning and
sometimes during training: because of the dynamic loss scaling strategy, there are points during training where the
gradients have overflown, and the loss scaling factor is reduced to avoid this happening again at the next step.

This means that you may update your learning rate scheduler when there was no update, which is fine in general, but may
have an impact when you have very little training data, or if the first learning rate values of your scheduler are very
important. In this case, you can skip the learning rate scheduler updates when the optimizer step was not done like
this:

```
if not accelerator.optimizer_step_was_skipped:
    lr_scheduler.step()
```

### Gradient Accumulation 

To perform gradient accumulation use [`~Accelerator.accumulate`] and specify a `gradient_accumulation_steps`. 
This will also automatically ensure the gradients are synced or unsynced when on multi-device training, check if the step should
actually be performed, and auto-scale the loss:

```python
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

for input, label in training_dataloader:
    with accelerator.accumulate(model):
        predictions = model(input)
        loss = loss_function(predictions, label)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### DeepSpeed

DeepSpeed support is experimental, so the underlying API will evolve in the near future and may have some slight
breaking changes. In particular, 🤗 Accelerate does not support DeepSpeed config you have written yourself yet, this
will be added in a next version.

<Tip warning={true}>

    The [`notebook_launcher`] does not support the DeepSpeed integration yet.

</Tip>

## Internal mechanism

Internally, the library works by first analyzing the environment in which the script is launched to determine which
kind of distributed setup is used, how many different processes there are and which one the current script is in. All
that information is stored in the [`~AcceleratorState`].

This class is initialized the first time you instantiate an [`~Accelerator`] as well as performing any
specific initialization your distributed setup needs. Its state is then uniquely shared through all instances of
[`~state.AcceleratorState`].

Then, when calling [`~Accelerator.prepare`], the library:

- wraps your model(s) in the container adapted for the distributed setup,
- wraps your optimizer(s) in a [`~optimizer.AcceleratedOptimizer`],
- creates a new version of your dataloader(s) in a [`~data_loader.DataLoaderShard`].

While the model(s) and optimizer(s) are just put in simple wrappers, the dataloader(s) are re-created. This is mostly
because PyTorch does not let the user change the `batch_sampler` of a dataloader once it's been created and the
library handles the sharding of your data between processes by changing that `batch_sampler` to yield every other
`num_processes` batches.

The [`~data_loader.DataLoaderShard`] subclasses `DataLoader` to add the following functionality:

- it synchronizes the appropriate random number generator of all processes at each new iteration, to ensure any
  randomization (like shuffling) is done the exact same way across processes.
- it puts the batches on the proper device before yielding them (unless you have opted out of
  `device_placement=True`).

The random number generator synchronization will by default synchronize:

- the `generator` attribute of a given sampler (like the PyTorch `RandomSampler`) for PyTorch >= 1.6
- the main random number generator in PyTorch <=1.5.1

You can choose which random number generator(s) to synchronize with the `rng_types` argument of the main
[`Accelerator`]. In PyTorch >= 1.6, it is recommended to rely on a local `generator` to avoid
setting the same seed in the main random number generator in all processes.

<Tip warning={true}>

    Synchronization of the main torch (or CUDA or XLA) random number generator will affect any other potential random
    artifacts you could have in your dataset (like random data augmentation) in the sense that all processes will get
    the same random numbers from the torch random modules (so will apply the same random data augmentation if it's
    controlled by torch).

</Tip>

<Tip>

    The randomization part of your custom sampler, batch sampler or iterable dataset should be done using a local
    `torch.Generator` object (in PyTorch >= 1.6), see the traditional `RandomSampler`, as an example.

</Tip>

For more details about the internals, see the [Internals page](package_reference/torch_wrappers).
