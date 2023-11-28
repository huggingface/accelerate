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

import functools
import logging
import os

from .state import PartialState


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    `log` takes in an additional `main_process_only` kwarg, which dictates whether it should be called on all processes
    or only the main executed one. Default is `main_process_only=True`.

    Does not require an `Accelerator` object to be created first.
    """

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        state = PartialState()
        return not main_process_only or (main_process_only and state.is_main_process)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed

        Also accepts "in_order", which if `True` makes the processes log one by one, in order. This is much easier to
        read, but comes at the cost of sometimes needing to wait for the other processes. Default is `False` to not
        break with the previous behavior.

        `in_order` is ignored if `main_process_only` is passed.
        """
        if PartialState._shared_state == {}:
            raise RuntimeError(
                "You must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility."
            )
        main_process_only = kwargs.pop("main_process_only", True)
        in_order = kwargs.pop("in_order", False)

        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

            elif in_order:
                state = PartialState()
                for i in range(state.num_processes):
                    if i == state.process_index:
                        msg, kwargs = self.process(msg, kwargs)
                        self.logger.log(level, msg, *args, **kwargs)
                    state.wait_for_everyone()

    @functools.lru_cache(None)
    def warning_once(self, *args, **kwargs):
        """
        This method is identical to `logger.warning()`, but will emit the warning with the same message only once

        Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the
        cache. The assumption here is that all warning messages are unique across the code. If they aren't then need to
        switch to another type of cache that includes the caller frame information in the hashing function.
        """
        self.warning(*args, **kwargs)


def get_logger(name: str, log_level: str = None):
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False` If a log should be called on all
    processes and in order, also pass `in_order=True`

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the `LOG_LEVEL` environment variable, or `INFO` if not

    Example:

    ```python
    >>> from accelerate.logging import get_logger
    >>> from accelerate import Accelerator

    >>> logger = get_logger(__name__)

    >>> accelerator = Accelerator()
    >>> logger.info("My log", main_process_only=False)
    >>> logger.debug("My log", main_process_only=True)

    >>> logger = get_logger(__name__, log_level="DEBUG")
    >>> logger.info("My log")
    >>> logger.debug("My second log")

    >>> array = ["a", "b", "c", "d"]
    >>> letter_at_rank = array[accelerator.process_index]
    >>> logger.info(letter_at_rank, in_order=True)
    ```
    """
    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, {})
