# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from pathlib import Path
from typing import List, Union
from unittest import mock

import torch

from ..state import AcceleratorState
from ..utils import gather, is_comet_ml_available, is_tensorflow_available, is_tpu_available, is_wandb_available


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def slow(test_case):
    """
    Decorator marking a test as slow. Slow tests are skipped by default. Set the RUN_SLOW environment variable to a
    truthy value to run them.
    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def require_cuda(test_case):
    """
    Decorator marking a test that requires CUDA. These tests are skipped when there are no GPU available.
    """
    return unittest.skipUnless(torch.cuda.is_available(), "test requires a GPU")(test_case)


def require_tpu(test_case):
    """
    Decorator marking a test that requires TPUs. These tests are skipped when there are no TPUs available.
    """
    return unittest.skipUnless(is_tpu_available(), "test requires TPU")(test_case)


def require_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup. These tests are skipped on a machine without multiple
    GPUs.
    """
    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)


def require_tensorflow(test_case):
    """
    Decorator marking a test that requires TensorFlow installed. These tests are skipped when TensorFlow isn't
    installed
    """
    return unittest.skipUnless(is_tensorflow_available(), "test requires TensorFlow")(test_case)


def require_wandb(test_case):
    """
    Decorator marking a test that requires wandb installed. These tests are skipped when wandb isn't installed
    """
    return unittest.skipUnless(is_wandb_available(), "test requires wandb")(test_case)


def require_comet_ml(test_case):
    """
    Decorator marking a test that requires comet_ml installed. These tests are skipped when comet_ml isn't installed
    """
    return unittest.skipUnless(is_comet_ml_available(), "test requires comet_ml")(test_case)


class TempDirTestCase(unittest.TestCase):
    """
    A TestCase class that keeps a single `tempfile.TemporaryDirectory` open for the duration of the class, wipes its
    data at the start of a test, and then destroyes it at the end of the TestCase.

    Useful for when a class or API requires a single constant folder throughout it's use, such as Weights and Biases

    The temporary directory location will be stored in `self.tmpdir`
    """

    clear_on_setup = True

    @classmethod
    def setUpClass(cls):
        "Creates a `tempfile.TemporaryDirectory` and stores it in `cls.tmpdir`"
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        "Remove `cls.tmpdir` after test suite has finished"
        if os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def setUp(self):
        "Destroy all contents in `self.tmpdir`, but not `self.tmpdir`"
        if self.clear_on_setup:
            for path in Path(self.tmpdir).glob("**/*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)


class MockingTestCase(unittest.TestCase):
    """
    A TestCase class designed to dynamically add various mockers that should be used in every test, mimicking the
    behavior of a class-wide mock when defining one normally will not do.

    Useful when a mock requires specific information available only initialized after `TestCase.setUpClass`, such as
    setting an environment variable with that information.

    The `add_mocks` function should be ran at the end of a `TestCase`'s `setUp` function, after a call to
    `super().setUp()` such as:
    ```python
    def setUp(self):
        super().setUp()
        mocks = mock.patch.dict(os.environ, {"SOME_ENV_VAR", "SOME_VALUE"})
        self.add_mocks(mocks)
    ```
    """

    def add_mocks(self, mocks: Union[mock.Mock, List[mock.Mock]]):
        """
        Add custom mocks for tests that should be repeated on each test. Should be called during
        `MockingTestCase.setUp`, after `super().setUp()`.

        Args:
            mocks (`mock.Mock` or list of `mock.Mock`):
                Mocks that should be added to the `TestCase` after `TestCase.setUpClass` has been run
        """
        self.mocks = mocks if isinstance(mocks, (tuple, list)) else [mocks]
        for m in self.mocks:
            m.start()
            self.addCleanup(m.stop)


def are_the_same_tensors(tensor):
    state = AcceleratorState()
    tensor = tensor[None].clone().to(state.device)
    tensors = gather(tensor).cpu()
    tensor = tensor[0].cpu()
    for i in range(tensors.shape[0]):
        if not torch.equal(tensors[i], tensor):
            return False
    return True


class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # note: there is a warning for a possible deadlock when using `wait` with huge amounts of data in the pipe
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # If it starts hanging, will need to switch to the following code. The problem is that no data
    # will be seen until it's done and if it hangs for example there will be no debug info.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: the timeout doesn't seem to make any difference here
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    return result
