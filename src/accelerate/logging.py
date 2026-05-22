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
from __future__ import annotations

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

    def process(self, msg, kwargs):
        msg, kwargs = super().process(msg, kwargs)

        # set `stacklevel` to exclude ourself in `Logger.findCaller()` while respecting user's choice
        kwargs.setdefault("stacklevel", 2)

        state = PartialState()
        msg = f"[RANK {state.process_index}] {msg}"
        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed

        Also accepts "in_order", which if `True` makes the processes log one by one, in order. This is much easier to
        read, but comes at the cost of sometimes needing to wait for the other processes. Default is `False` to not
        break with the previous behavior.

        `main_process_only` is ignored if `in_order` is passed.
        """
        if PartialState._shared_state == {}:
            raise RuntimeError(
                "You must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility."
            )
        main_process_only = kwargs.pop("main_process_only", True)
        in_order = kwargs.pop("in_order", False)

        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            if not in_order and self._should_log(main_process_only):
                self.logger.log(level, msg, *args, **kwargs)

            elif in_order:
                state = PartialState()
                for i in range(state.num_processes):
                    if i == state.process_index:
                        self.logger.log(level, msg, *args, **kwargs)
                    state.wait_for_everyone()

    def warning_once(self, msg, *args, **kwargs):
        """
        Like `warning()`, but emits each unique message only once per adapter.

        The cache is keyed on the message text, so passing the standard
        `extra={...}` kwarg (or any other unhashable value) doesn't crash
        the way it did with the previous `lru_cache` decorator.
        """
        cache = getattr(self, "_warning_once_cache", None)
        if cache is None:
            cache = set()
            self._warning_once_cache = cache
        if msg in cache:
            return
        cache.add(msg)
        self.warning(msg, *args, **kwargs)


def get_logger(name: str, log_level: str | None = None):
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
