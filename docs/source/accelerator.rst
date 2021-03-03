.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Accelerator
=======================================================================================================================

The :class:`~accelerate.Accelerator` is the main class provided by 🤗 Accelerate. It serves at the main entrypoint for
the API. To quickly adapt your script to work on any kind of setup with 🤗 Accelerate juste:

1. Initialize an :class:`~accelerate.Accelerator` object (that we will call :obj:`accelerator` in the rest of this
   page) as early as possible in your script.
2. Pass along your model(s), optimizer(s), dataloader(s) to the :meth:`~accelerate.Accelerator.prepare` method.
3. (Optional but best practice) Remove all the :obj:`.cuda()` or :obj:`.to(device)` in your code and let the
   :obj:`accelerator` handle device placement for you.
4. Replace the :obj:`loss.backward()` in your code by :obj:`accelerator.backward(loss)`.
5. (Optional, when using distributed evaluation) Gather your predictions and labelsbefore storing them or using them
   for metric computation using :meth:`~accelerate.Accelerator.gather`.

This is all what is needed in most cases. For more advanced case or a nicer experience here are the functions you
should search for and replace by the corresponding methods of your :obj:`accelerator`:

- :obj:`print` statements should be replaced by :meth:`~accelerate.Accelerator.print` to be only printed once per
  process.
- Use :meth:`~accelerate.Accelerator.is_local_main_process` for statements that should be executed once per server.
- Use :meth:`~accelerate.Accelerator.is_main_process` for statements that should be executed once only.
- Use :meth:`~accelerate.Accelerator.wait_for_everyone` to make sure all processes join that point before continuing
  (useful before a model save for instance).
- Use :meth:`~accelerate.Accelerator.unwrap_model` to unwrap your model before saving it.
- Use :meth:`~accelerate.Accelerator.save` instead of :obj:`torch.save`.
- Use :meth:`~accelerate.Accelerator.clip_grad_norm_` instead of :obj:`torch.nn.utils.clip_grad_norm_` and
  :meth:`~accelerate.Accelerator.clip_grad_value_` instead of :obj:`torch.nn.utils.clip_grad_value_`.

.. autoclass:: accelerate.Accelerator
    :members:
