.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Quick tour
=======================================================================================================================

Let's have a look at a look at 🤗 Accelerate main features and traps to avoid.

Main use
-----------------------------------------------------------------------------------------------------------------------

To use 🤗 Accelerate in your own script, you have to change four things:

1. Import the :class:`~accelerate.Accelerator` main class instantiate one in an :obj:`accelerator` object:

.. code-block:: python

    from accelerate import Accelerator

    accelerator = Accelerator()

This should happen as early as possible in your training script as it will initialize everything necessary for
distributed training. You don't need to indicate the kind of environment you are in (just one machine with a GPU, one
match with several GPUs, several machines with multiple GPUs or a TPU), the library will detect this automatically.

2. Remove the call :obj:`.to(device)` or :obj:`.cuda()` for your model and input data. The :obj:`accelerator` object
will handle this for you and place all those objects on the right device for you. If you know what you're doing, you
can leave those :obj:`.to(device)` calls but you should use the device provided by the :obj:`accelerator` object:
:obj:`accelerator.device`.

To fully deactivate the automatic device placement, pass along :obj:`device_placement=False` when initializing your
:class:`~accelerate.Accelerator`.

.. Warning::

    If you place your objects manually on the proper device, be careful to create your optimizer after putting your
    model on :obj:`accelerator.device` or your training will fail on TPU.

3. Pass all objects relevant to training (optimizer, model, training dataloader) to the
:meth:`~accelerate.Accelerator.prepare` method. This will make sure everything is ready for training.

.. code-block:: python

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

In particular, your training dataloader will be sharded accross all GPUs/TPU cores available so that each one sees a
different portion of the training dataset. Also, the random states of all processes will be synchronized at the
beginning of each iteration through your dataloader, to make sure the data is shuffled the same way (if you decided to
use :obj:`shuffle=True` or any kind of random sampler).

.. Note::

    The actual batch size for your training will be the number of devices used multiplied by the batch size you set in
    your script: for instance training on 4 GPUs with a batch size of 16 set when creating the training dataloader will
    train at an actual batch size of 64.

    Alternatively, you can use the option :obj:`split_batches=True` when creating initializing your
    :class:`~accelerate.Accelerator`, in which case the batch size will always stay the same, whether your run your
    script on 1, 2, 4 or 64 GPUs.

You should execute this instruction as soon as all objects for training are created, before starting your actual
training loop.

.. Warning::

    Your training dataloader may change length when going through this method: if you run on X GPUs, it will have its
    length divided by X (since your actual batch size will be multiplied by X), unless you set
    :obj:`split_batches=True`.

    Any instruction using your training dataloader length (for instance if you need the number of total training steps
    to create a learning rate scheduler) should go after the call to :meth:`~accelerate.Accelerator.prepare`.

You can perfectly send your dataloader to :meth:`~accelerate.Accelerator.prepare` on its own, but it's best to send the
model and optimizer to :meth:`~accelerate.Accelerator.prepare` together.

You may or may not want to send your validation dataloader to :meth:`~accelerate.Accelerator.prepare`, depending on
whether you want to run distributed evaluation or not (see below).

4. Replace the line :obj:`loss.backward()` by :obj:`accelerator.backward(loss)`.

And you're all set! With all these changes, your script will run on your local machine as well as on multiple GPUs or a
TPU! You can either use your favorite tool to launch the distributed training, or you can use the 🤗 Accelerate
launcher.


Distributed evaluation
-----------------------------------------------------------------------------------------------------------------------

You can perform regular evaluation in your training script, if you leave your validation dataloader out of the
:meth:`~accelerate.Accelerator.prepare` method. In this case, you will need to put the input data on the
:obj:`accelerator.device` manually.

To perform distributed evaluation, send along your validation dataloader to the :meth:`~accelerate.Accelerator.prepare`
method:

.. code-block:: python

    validation_dataloader = accelerator.prepare(validation_dataloader)

Like for your training dataloader, it will mean that (should you run your script on multiple devices) each device will
only see part of the evaluation data. This means you will need to group your predictions together. This is very easy to
do with the :meth:`~accelerate.Accelerator.gather` method.

.. code-block:: python

    for inputs, targets in validation_dataloader:
        predictions = model(inputs)
        # Gather all predictions and targets
        all_predictions = accelerator.gather(predictions)
        all_targets = accelerator.gather(targets)
        # Example of use with a `Datasets.Metric`
        metric.add_batch(all_predictions, all_targets)


.. Warning::

    Like for the training dataloader, passing your validation dataloader through
    :meth:`~accelerate.Accelerator.prepare` may change its: if you run on X GPUs, it will have its length divided by X
    (since your actual batch size will be multiplied by X), unless you set :obj:`split_batches=True`.

    Any instruction using your training dataloader length (for instance if you need the number of total training steps
    to create a learning rate scheduler) should go after the call to :meth:`~accelerate.Accelerator.prepare`.


Launching your distributed script
-----------------------------------------------------------------------------------------------------------------------

You can use the regular commands to launch your distributed training (like :obj:`torch.distributed.launch` for
PyTorch), they are fully compatible with 🤗 Accelerate. The only caveat here is that 🤗 Accelerate uses the environment
to determine all useful information, so :obj:`torch.distributed.launch` should be used with the flag :obj:`--use_env`.

🤗 Accelerate also provides a CLI tool that unifies all launcher, so you only have to remember one command. To use it,
just run

.. code-block:: bash

    accelerate config

on your machine and reply to the questions asked. This will save a `default_config.json` file in your cache folder for
🤗 Accelerate. That cache folder is (with decreasing order of priority):

    - The content of your environment variable ``HF_HOME`` suffixed with `accelerate`.
    - If it does not exist, the content of your environment variable ``XDG_CACHE_HOME`` suffixed with
      `huggingface/accelerate`.
    - If this does not exist either, the folder `~/.cache/huggingface/accelerate`

You can also specify with the flag :obj:`--config_file` the location of the file you want to save.

Once this is done, you can test everything is going well on your setup by running

.. code-block:: bash

    accelerate test


This will launch a short script that will test the distributed environment. If it runs fine, you are ready for the next
step!

Note that if you specified a location for the config file in the previous step, you need to pass it here as well:

.. code-block:: bash

    accelerate test --config_file path_to_config.json


Now that this is done, you can run your script with the following command:

.. code-block:: bash

    accelerate launch path_to_script.py --args_for_the_script


If you stored the config file in a non-default location, you can indicate it to the launcher like his:

.. code-block:: bash

    accelerate launch --config_file path_to_config.json path_to_script.py --args_for_the_script

You can also override any of the arguments determined by your config file, see TODO: insert ref here.


Training on TPU
-----------------------------------------------------------------------------------------------------------------------

If you want to launch your script on TPUs, there are a few caveats you should be aware of. Behind the scenes, the TPUs
will create a graph of all the operations happening im your training step (forward pass, backward pass and optimizer
step). This is why your first step of training will always be very long as building and compiling this graph for
optimizations takes some time.

The good news is that this compilation will be cached so the second step and all the following will be much faster. The
bas news is that it only applies if all of your steps do exactly the same operations, which implies:

- having all tensors of the same length in all your lenghts
- having static code (i.e., not a foor loop of length that could change from step to step)

Having any of the things above change between two steps will trigger a new compilation which will, once again, take a
lof of time. In practice, that means you must take special care to have all your tensors in your inputs of the same
shape (so no dynamic padding for instance if you are in an NLP problem) and should not use layer with for loops that
have different lengths depending on the inputs (such as an LSTM) or the training will be excruciatingly slow.

To introduce special behavior in your script for TPUs you can check the :obj:`distributed_type` of your :obj:`accelerator`:

.. code-block:: python

    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.TPU:
        # do something of static shape
    else:
        # go crazy and be dynamic

The `NLP example <https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py>`__ shows an example in
situation with dynamic padding.


Other caveats
-----------------------------------------------------------------------------------------------------------------------

We list here all smaller issues you could have in your script conversion and how to resolve them.

Execute a statement only on one processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of your instructions only need to run for one process on a given server: for instance a data download or a log
statement. To do this, wrap the statement in a test like this:

.. code-block:: python

    if accelerator.is_local_main_process:
        # Is executed once per server

Another example is progress bars: to avoid having multiple progress bars in your output, you should only display one on
the local main process:

.. code-block:: python

    from tqdm.auto import tqdm

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

The `local` means per machine: if you are running your training on two servers with several GPUs, the instruction will
be executed once on each of those servers. If you need to execute something only once for all processes (and not per
machine) for instance, uploading the final model to the 🤗 model hub, wrap it in a test like this:

.. code-block:: python

    if accelerator.is_main_process:
        # Is executed once only

For printing statements you only want executed once per machine, you can just replace the :obj:`print` function by
:obj:`accelerator.print`.


Defer execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you run your usual script, instructions are executed in order. Using 🤗 Accelerate to deploy your script on several
GPUs at the same time introduces a complication: while each process executes all instructions in order, some may be
faster than others.

You might need to wait for all processes to have reached a certain point before executing a given instruction. For
instance, you shouldn't save a model before being sure every process is done with training. To do this, just write the
following line in your code:

.. code-block::

    accelerator.wait_for_everyone()

This instruction will block all the processes that arrive them first until all the other processes have reached that
point (if you run your script on just one GPU or CPU, this wont' do anything).


Saving/loading a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saving the model you trained might need a bit of adjustment: first you should wait for all processes to reach that
point in the script as shown above, and then, you should unwrap your model before saving it. This is because when going
through the :meth:`~accelerate.Accelerator.prepare` method, your model may have been placed inside a bigger model,
which deals with the distributed training. This in turn means that saving your model state dictionary without taking
any precaution will take that potential extra layer into account, and you will end up with weights you can't load back
in your base model.

This is why it's recommended to `unwrap` your model first. Here is an example:

.. code-block::

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), filename)

If your script contains a logic to load checkpoint, we also recommend you load your weights in the unwrapped model
(this is only useful if you use the load function after making your model go through
:meth:`~accelerate.Accelerator.prepare`). Here is an example:

.. code-block::

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(filename))

Note that since all the model parameters are references to tensors, this will load your weights inside :obj:`model`.

Gradient clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using gradient clipping in your script, you should replace the calls to
:obj:`torch.nn.utils.clip_grad_norm_` or :obj:`torch.nn.utils.clip_grad_value_` with :obj:`accelerator.clip_grad_norm_`
and :obj:`accelerator.clip_grad_value_` respectively.


Internal mechanism
-----------------------------------------------------------------------------------------------------------------------

Internally, the library works by first analyzing the environment in which the script is launched to determine which
kind of distributed setup is used, how many different processes there are and which one the current script is in. All
that information is stored in the :class:`~accelerate.state.AcceleratorState`.

This class is initialized the first time you instantiate a :class:`~accelerate.Accelerator` as well as performing any
specific initialization your distributed setup needs. Its state is then uniquely shared through all instances of
:class:`~accelerate.state.AcceleratorState`.

Then, when calling :meth:`~accelerate.Accelerator.prepare`, the library:

- wraps your model(s) in the container adapted for the distributed setup,
- wraps your optimizer(s) in a :class:`~accelerate.optimizer.AcceleratedOptimizer`,
- creates a new version of your dataloader(s) in a :class:`~accelerate.data_loader.DataLoaderShard`.

While the model(s) and optimizer(s) are just put in simple wrappers, the dataloader(s) are re-created. This is mostly
because PyTorch does not let the user change the :obj:`batch_sampler` of a dataloader once it's been created and the
library handles the sharding of your data between processes by changing that :obj:`batch_sampler` to yield every other
:obj:`num_processes` batches.

The :class:`~accelerate.data_loader.DataLoaderShard` subclasses :obj:`DataLoader` to add the following functionality:

- it synchronizes the torch random number generators of all processes at each new iteration, to ensure any
  randomization (like shuffling) is done the exact same way across processes.
- it puts the batches on the proper device before yielding them (unless you have opted out of
  :obj:`device_placement=True`).

.. Warning::

    The random number generator synchronization will affect any other potential random artifacts you could have in your
    dataset (like random data augmentation) in the sense all processes will get the same random numbers from the torch
    random modules (so will apply the same random data augmentation if it's controlled by torch). While this is usually
    fine, you should use the random number generator from the Python :obj:`random` module or NumPy for your data
    augmentation if you think this will be a problem.

    The randomization part of your sampler on the other hand should absolutely be done using the torch random number
    generator (like in the traditional :obj:`RandomSampler`).

See more details about the internal in the :doc:`Internals page <internal>`.
