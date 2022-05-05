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

import logging

from .state import AcceleratorState


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    `log` takes in an additional `main_process_only` kwarg, which dictates whether it should be called on all processes
    or only the main executed one. Default is `main_process_only=True`.
    """

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        return not main_process_only or (main_process_only and AcceleratorState().local_process_index == 0)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed
        """
        main_process_only = kwargs.pop("main_process_only", True)
        if self.isEnabledFor(level) and self._should_log(main_process_only):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str):
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False`

    E.g.
    ```python
    logger.info("My log", main_process_only=False)
    logger.debug("My log", main_process_only=False)
    ```

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
    """
    logger = logging.getLogger(name)
    return MultiProcessAdapter(logger, {})
