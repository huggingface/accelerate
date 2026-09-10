# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Lightweight callback system for `accelerate`.

Callbacks receive lifecycle events fired by `Accelerator` methods, and can be used to add custom logic (logging,
early stopping, scheduling, ...) without modifying the training script:

```python
>>> from accelerate import Accelerator, Callback

>>> class EarlyStopping(Callback):
...     def on_step_end(self, accelerator, loss=None, **kwargs):
...         if loss is not None and loss > 10.0:
...             accelerator.set_trigger()

>>> accelerator = Accelerator(callbacks=[EarlyStopping()])
```
"""

from .logging import get_logger


__all__ = ["Callback", "CallbackHandler"]

logger = get_logger(__name__)

CALLBACK_EVENTS = (
    "on_init_end",
    "on_train_begin",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_step_begin",
    "on_step_end",
    "on_optimizer_step",
    "on_backward_end",
    "on_log",
    "on_save",
    "on_save_end",
)


class Callback:
    """
    Base class for all accelerate callbacks. Subclass it and override the events you need; every method receives the
    firing `Accelerator` instance as its first positional argument, so a single callback class works unchanged on any
    distributed setup.

    Available events (all no-ops by default):

    - `on_init_end(accelerator, **kwargs)` -- after `Accelerator.__init__` finishes
    - `on_train_begin(accelerator, **kwargs)` -- at the start of training (call manually)
    - `on_train_end(accelerator, **kwargs)` -- at the end of training (call manually or via `end_training`)
    - `on_epoch_begin(accelerator, epoch=None, **kwargs)` -- at the start of each epoch (call manually)
    - `on_epoch_end(accelerator, epoch=None, **kwargs)` -- at the end of each epoch (call manually)
    - `on_step_begin(accelerator, batch=None, **kwargs)` -- before a batch is processed
    - `on_step_end(accelerator, loss=None, **kwargs)` -- after a batch is processed; receives the loss if the user
      passes it to `accelerator.callback_step_end(loss=...)` or calls `on_step_end` manually
    - `on_backward_end(accelerator, **kwargs)` -- after `accelerator.backward()` completes
    - `on_optimizer_step(accelerator, **kwargs)` -- after a prepared optimizer actually steps (grad-sync steps only)
    - `on_log(accelerator, values=None, step=None, **kwargs)` -- when `accelerator.log()` is called
    - `on_save(accelerator, output_dir=None, **kwargs)` -- before `accelerator.save_state()`
    - `on_save_end(accelerator, output_dir=None, **kwargs)` -- after `accelerator.save_state()` completes

    Callbacks run on **all** processes by default. Wrap logic with `accelerator.on_main_process` (or the other
    process decorators) to restrict where it runs:

    ```python
    >>> class LogLoss(Callback):
    ...     def on_step_end(self, accelerator, loss=None, **kwargs):
    ...         accelerator.print(f"step {accelerator.step}: loss={loss}")
    ```

    Example of a full training loop with callbacks:

    ```python
    >>> accelerator = Accelerator(gradient_accumulation_steps=4, callbacks=[MyCallback()])
    >>> model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    >>> accelerator.trigger_callbacks("on_train_begin")
    >>> for epoch in range(num_epochs):
    ...     accelerator.trigger_callbacks("on_epoch_begin", epoch=epoch)
    ...     for batch in dataloader:
    ...         with accelerator.accumulate(model):
    ...             outputs = model(batch)
    ...             loss = loss_fn(outputs, targets)
    ...             accelerator.backward(loss)
    ...             accelerator.callback_step_begin(batch=batch)
    ...             optimizer.step()
    ...             optimizer.zero_grad()
    ...         accelerator.callback_step_end(loss=loss)
    ...     accelerator.trigger_callbacks("on_epoch_end", epoch=epoch)
    >>> accelerator.trigger_callbacks("on_train_end")
    ```
    """

    def on_init_end(self, accelerator, **kwargs):
        pass

    def on_train_begin(self, accelerator, **kwargs):
        pass

    def on_train_end(self, accelerator, **kwargs):
        pass

    def on_epoch_begin(self, accelerator, epoch=None, **kwargs):
        pass

    def on_epoch_end(self, accelerator, epoch=None, **kwargs):
        pass

    def on_step_begin(self, accelerator, batch=None, **kwargs):
        pass

    def on_step_end(self, accelerator, loss=None, **kwargs):
        pass

    def on_backward_end(self, accelerator, **kwargs):
        pass

    def on_optimizer_step(self, accelerator, **kwargs):
        pass

    def on_log(self, accelerator, values=None, step=None, **kwargs):
        pass

    def on_save(self, accelerator, output_dir=None, **kwargs):
        pass

    def on_save_end(self, accelerator, output_dir=None, **kwargs):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


class CallbackHandler:
    """
    Internal helper owned by `Accelerator` that stores the registered callbacks and dispatches events to them.

    Args:
        callbacks (list of [`~callbacks.Callback`], *optional*):
            The callbacks to register at creation time. More can be added later with `add_callback` /
            `remove_callback`.
    """

    def __init__(self, callbacks=None):
        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]:
                self.add_callback(cb)

    def add_callback(self, callback):
        """Register a callback. Passing a class (not an instance) is supported and will be instantiated."""
        if isinstance(callback, type):
            callback = callback()
        if not isinstance(callback, Callback):
            raise TypeError(f"The callback passed to add_callback must be a `Callback` instance, got {callback!r}")
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a registered callback (by instance or by class)."""
        if isinstance(callback, type):
            callback = next((cb for cb in self.callbacks if isinstance(cb, callback)), None)
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def __len__(self):
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def __repr__(self):
        return f"CallbackHandler(callbacks={[type(cb).__name__ for cb in self.callbacks]})"

    def call_event(self, event_name, accelerator, **kwargs):
        """Fire `event_name` on every registered callback, isolating failures per callback."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(accelerator, **kwargs)
            except Exception:
                logger.exception(f"Callback {callback!r} raised an exception in `{event_name}`, continuing.")
