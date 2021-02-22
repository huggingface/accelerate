Accelerate
=======================================================================================================================

Run your *raw* PyTorch training script on any kind of device

Features
-----------------------------------------------------------------------------------------------------------------------

- 🤗 Accelerate provides an easy API to make your scripts run with mixed precision and on any kind of distributed
  setting (multi-GPUs, TPUs etc.) while still letting you write your own training loop. The same code can then runs
  seamlessly on your local machine for debugging or your training environment.

- 🤗 Accelerate also provides a CLI tool that allows you to quickly configure and test your training environment then
  launch the scripts.


Easy to integrate
-----------------------------------------------------------------------------------------------------------------------

A traditional training loop in PyTorch looks like this:

.. code-block:: python

    my_model.to(device)

    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        loss.backward()
        my_optimizer.step()

Changing it to work with accelerate is really easy and only adds a few lines of code:

.. code-block:: python

    from accelerate import Accelerator

    accelerator = Accelerator(device_placement=False)
    # Use the device given by the `accelerator` object.
    device = accelerator.device
    my_model.to(device)
    # Pass every important object (model, optimizer, dataloader) to `accelerator.prepare`
    my_model, my_optimizer, my_training_dataloader = accelerate.prepare(
        my_model, my_optimizer, my_training_dataloader
    )

    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        # Just a small change for the backward instruction
        accelerate.backward(loss)
        my_optimizer.step()

and with this, your script can now run in a distributed environment (multi-GPU, TPU).

You can even simplify your script a bit by letting 🤗 Accelerate handle the device placement for you (which is safer, especially for TPU training):

.. code-block:: python

    from accelerate import Accelerator

    accelerator = Accelerator()
    # Pass every important object (model, optimizer, dataloader) to `accelerator.prepare`
    my_model, my_optimizer, my_training_dataloader = accelerate.prepare(
        my_model, my_optimizer, my_training_dataloader
    )

    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        # Just a small change for the backward instruction
        accelerate.backward(loss)
        my_optimizer.step()


Script launcher
-----------------------------------------------------------------------------------------------------------------------

No need to remember how to use ``torch.distributed.launch`` or to write a specific launcher for TPU training! 🤗 
Accelerate comes with a CLI tool that will make your life easier when launching distributed scripts.

On your machine(s) just run:

.. code-block:: bash

    accelerate config

and answer the questions asked. This will generate a config file that will be used automatically to properly set the
default options when doing

.. code-block:: bash

    accelerate launch my_script.py --args_to_my_script

For instance, here is how you would run the GLUE example on the MRPC task (from the root of the repo):

.. code-block:: bash

    accelerate launch examples/glue_example.py --task_name mrpc --model_name_or_path bert-base-cased

.. toctree::
    :maxdepth: 2
    :caption: Get started

    quicktour
    installation


.. toctree::
    :maxdepth: 2
    :caption: API reference

    accelerator
