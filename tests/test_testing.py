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

import signal
import sys

import pytest

from accelerate.test_utils.testing import execute_subprocess_async


def test_subprocess_success_preserves_output():
    result = execute_subprocess_async(
        [sys.executable, "-c", "import sys; print('output'); print('diagnostic', file=sys.stderr)"],
        quiet=True,
        echo=False,
    )

    assert result.returncode == 0
    assert result.stdout == ["output"]
    assert result.stderr == ["diagnostic"]


def test_subprocess_nonzero_exit_raises_with_stderr():
    with pytest.raises(RuntimeError, match="returncode 3") as error:
        execute_subprocess_async(
            [sys.executable, "-c", "import sys; print('failure diagnostic', file=sys.stderr); sys.exit(3)"],
            quiet=True,
            echo=False,
        )

    assert "failure diagnostic" in str(error.value)


@pytest.mark.skipif(sys.platform == "win32", reason="Negative signal return codes require POSIX")
def test_subprocess_signal_exit_raises_with_stderr():
    with pytest.raises(RuntimeError, match=f"returncode {-signal.SIGTERM}") as error:
        execute_subprocess_async(
            [
                sys.executable,
                "-c",
                "import os, signal, sys; "
                "print('signal diagnostic', file=sys.stderr, flush=True); "
                "os.kill(os.getpid(), signal.SIGTERM)",
            ],
            quiet=True,
            echo=False,
        )

    assert "signal diagnostic" in str(error.value)
