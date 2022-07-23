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

from contextlib import contextmanager

import torch

from rich.console import Console

from ..state import get_int_from_env
from .imports import is_tpu_available


if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


def _is_local_main_process():
    if is_tpu_available():
        return xm.get_local_ordinal() == 0
    elif torch.distributed.is_initialized():
        return (
            get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
            )
            == 0
        )
    else:
        return True


@contextmanager
def clean_traceback(show_locals: bool = False):
    """
    A context manager that uses `rich` to provide a clean traceback when dealing with multiprocessed logs.

    Args:
        show_locals (`bool`, *optional*, defaults to False):
            Whether to show local objects as part of the final traceback
    """

    console = Console()
    try:
        yield
    except:
        if _is_local_main_process():
            console.print_exception(suppress=[__file__], show_locals=show_locals)
