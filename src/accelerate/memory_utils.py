# Copyright 2022 The HuggingFace Team. All rights reserved.
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
A collection of utilities for ensuring that training can always occur. Heavily influenced by the
[toma](https://github.com/BlackHC/toma) library.
"""

import functools
import inspect


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def memory_aware(
    function: callable = None,
    starting_batch_size: int = 128,
    function_arg_name: str = "dataloaders",
    dataloader_function: callable = None,
    dataloader_arg_name: str = "batch_size",
    dataloader_function_kwargs: dict = {},
):
    """
    A decorator that tries to execute `function`. If the wrapped function fails from any exceptions related to
    out-of-memory or CUDNN not supported, the batch size is cut in half, new dataloaders are built, and the function is
    ran again until it executes completely.

    If the `DataLoader(s)` are built outside of `function`, a `dataloader_function` can be passed with optional
    `dataloader_function_kwargs`. These will then be passed into `function` as `arg_name`

    Args:
        function (`callable`, *optional*):
            A function to wrap that utilizes a batch size or dataloaders
        starting_batch_size (`int`, *optional*, defaults to 128):
            The initial batch size to fit with.
        function_arg_name(`str`, *optional*):
            The argument in `function` to pass in generated `dataloaders` or the active batch size
        dataloader_function (`callable`, *optional*):
            An optional generator that builds PyTorch `DataLoaders`.
        dataloader_arg_name (`str`, *optional*, defaults to "batch_size"):
            The name of the argument to pass to `dataloader_function` to specify the batch_size parameter
        dataloader_function_kwargs (`dict`, *optional*):
            Optional kwargs that get passed to `dataloader_function`. Should not include `function_arg_name`.
    """
    if function_arg_name in dataloader_function_kwargs:
        raise ValueError(
            f"`dataloader_function_kwargs` should not contain {function_arg_name} as it is changed and passed dynamically."
        )
    if function is None:
        if dataloader_function is None:
            return functools.partial(
                memory_aware, starting_batch_size=starting_batch_size, function_arg_name=function_arg_name
            )
        else:
            return functools.partial(
                memory_aware,
                starting_batch_size=starting_batch_size,
                function_arg_name=function_arg_name,
                dataloader_function=dataloader_function,
                dataloader_arg_name=dataloader_arg_name,
                dataloader_function_kwargs=dataloader_function_kwargs,
            )

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        # Access our `batch_size`
        nonlocal batch_size
        args = list(args)
        param_name_to_idx = {param: i for i, param in enumerate(inspect.signature(function).parameters)}
        # We need to know what `arg` is our `batch_size` argument
        arg_idx = param_name_to_idx[function_arg_name]
        while True:
            try:
                if dataloader_function is not None:
                    dataloader_function_kwargs[dataloader_arg_name] = batch_size
                    dataloaders = dataloader_function(**dataloader_function_kwargs)
                    args[arg_idx] = dataloaders
                    return function(*args, **kwargs)
                else:
                    args[arg_idx] = batch_size
                    return function(*args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    batch_size /= 2
                else:
                    raise

    return decorator
