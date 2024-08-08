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

import warnings

from .imports import is_tqdm_available


if is_tqdm_available():
    from tqdm.auto import tqdm as _tqdm

from ..state import PartialState


def tqdm(*args, main_process_only: bool = True, **kwargs):
    """
    Wrapper around `tqdm.tqdm` that optionally displays only on the main process.

    Args:
        main_process_only (`bool`, *optional*):
            Whether to display the progress bar only on the main process
    """
    if not is_tqdm_available():
        raise ImportError("Accelerate's `tqdm` module requires `tqdm` to be installed. Please run `pip install tqdm`.")
    if len(args) > 0 and isinstance(args[0], bool):
        warnings.warn(
            f"Passing `{args[0]}` as the first argument to Accelerate's `tqdm` wrapper is deprecated "
            "and will be removed in v0.33.0. Please use the `main_process_only` keyword argument instead.",
            FutureWarning,
        )
        main_process_only = args[0]
        args = args[1:]
    disable = kwargs.pop("disable", False)
    if main_process_only and not disable:
        disable = PartialState().local_process_index != 0
    return _tqdm(*args, **kwargs, disable=disable)
