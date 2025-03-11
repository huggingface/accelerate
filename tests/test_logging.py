# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import logging
import os

import pytest

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState


def current_lineno() -> int:
    # A simple helper that returns the lineno of its call-site.
    caller_frame = inspect.currentframe().f_back
    caller_info = inspect.getframeinfo(caller_frame)
    return caller_info.lineno


class CustomLogger(logging.LoggerAdapter):
    # Mocks a user-defined custom logger wrapper that sets `stacklevel=3`.
    def log(self, level, msg, *args, **kwargs):
        # E.g. the user wants to modify `stacklevel`, `accelerate.logging`
        # should respect the user's `stacklevel`. For the specific value
        # of `3`, calling `CustomLogger.log()`, etc., should log that callsite,
        # rather than the callsite of the following `self.logger.log()`.
        kwargs["stacklevel"] = 3
        self.logger.log(level, msg, *args, **kwargs)


@pytest.fixture(scope="module")
def accelerator():
    accelerator = Accelerator()
    yield accelerator
    AcceleratorState._reset_state(True)


@pytest.mark.usefixtures("accelerator")
def test_log_stack(caplog):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(filename)s:%(name)s:%(lineno)s:%(funcName)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    message = "Test"
    lineno = current_lineno() + 1  # the next line is the actual callsite
    logger.warning(message)

    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelname == logging.getLevelName(logging.WARNING)
    assert rec.filename == os.path.basename(__file__)
    assert rec.name == __name__
    assert rec.lineno == lineno
    assert rec.funcName == test_log_stack.__name__
    assert rec.message == message


@pytest.mark.usefixtures("accelerator")
def test_custom_stacklevel(caplog):
    wrapped_logger = get_logger(__name__)
    logging.basicConfig(
        format="%(filename)s:%(name)s:%(lineno)s:%(funcName)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    logger = CustomLogger(wrapped_logger, {})

    message = "Test"
    lineno = current_lineno() + 1  # the next line is the actual callsite
    logger.warning(message)

    # `CustomLogger.log` set custom `stacklevel=3`, so `logger.warning` should
    # log its callsite (rather than those of the `warpped_logger`).
    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelname == logging.getLevelName(logging.WARNING)
    assert rec.filename == os.path.basename(__file__)
    assert rec.name == __name__
    assert rec.lineno == lineno
    assert rec.funcName == test_custom_stacklevel.__name__
    assert rec.message == message
