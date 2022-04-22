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
from typing import List, Union

from torch.utils import DataLoader

from accelerate.data_loader import DataLoaderDispatcher, DataLoaderShard


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
    function: callable = None, dataloaders: List[Union[DataLoader, DataLoaderDispatcher, DataLoaderShard]] = None
):
    """
    A decorator that will reduce the batch size in half of all `dataloaders` if the wrapped function fails from any
    exceptions related to out-of-memory or CUDNN not supported until the function executes completely.

    Args:
        function (`callable`, *optional*):
            A function to wrap that utilizes all declared `dataloaders`
        dataloaders (list of `DataLoader`, [`~data_loader.DataLoaderDispatcher`], or [`~data_loader.DataLoaderShard`], *optional*):
            A list of prepared or unprepared `DataLoaders`.
    """
    if not isinstance(dataloaders, (tuple, list)):
        dataloaders = [dataloaders]
    if not all(
        (isinstance(dataloader, (DataLoaderDispatcher, DataLoaderShard, DataLoader)) for dataloader in dataloaders)
    ):
        raise TypeError(
            "Unsupported operation attempted. One or more dataloaders passed were not of type(s) `DataLoaderDispatcher`, `DataLoaderShard`, or `torch.utils.DataLoader`"
        )

    if function is None:
        return functools.partial(memory_aware, dataloaders=dataloaders)

    def decorator(*args, **kwargs):
        while True:
            try:
                return function(*args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    for dataloader in dataloaders:
                        dataloader.batch_size /= 2
                else:
                    raise

    return decorator
