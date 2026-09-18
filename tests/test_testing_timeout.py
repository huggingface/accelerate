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

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from accelerate.test_utils import testing
from accelerate.test_utils.testing import _stream_subprocess, execute_subprocess_async


@pytest.fixture
def child_processes(tmp_path):
    pid_file = tmp_path / "pids.json"
    yield pid_file

    # Clean up even when exercising the broken implementation or a failed assertion.
    if pid_file.exists():
        for pid in json.loads(pid_file.read_text()):
            try:
                process = psutil.Process(pid)
                process.kill()
                process.wait(timeout=2)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass


def assert_processes_stopped(pid_file):
    for pid in json.loads(pid_file.read_text()):
        try:
            process = psutil.Process(pid)
            assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


@pytest.mark.parametrize("timeout", [None, 5])
def test_subprocess_drains_both_output_streams(timeout):
    result = execute_subprocess_async(
        [
            sys.executable,
            "-c",
            "import sys; "
            "[print('out' * 100) for _ in range(1000)]; "
            "[print('err' * 100, file=sys.stderr) for _ in range(1000)]",
        ],
        timeout=timeout,
        quiet=True,
        echo=False,
    )

    assert result.returncode == 0
    assert result.stdout == ["out" * 100] * 1000
    assert result.stderr == ["err" * 100] * 1000


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize("close_output", [False, True])
def test_subprocess_timeout_stops_child(child_processes, close_output):
    code = (
        "import json, os, sys, time; from pathlib import Path; "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid()])); "
        "print('partial output', flush=True); "
        "print('partial diagnostic', file=sys.stderr, flush=True); "
    )
    if close_output:
        code += "os.close(1); os.close(2); "
    code += "time.sleep(5)"

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out") as error:
        execute_subprocess_async([sys.executable, "-c", code], timeout=1, quiet=True, echo=False)

    assert time.monotonic() - started < 4
    assert "partial output" in str(error.value)
    assert "partial diagnostic" in str(error.value)
    assert_processes_stopped(child_processes)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_subprocess_timeout_stops_descendants(child_processes, parent_exits):
    worker = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)"
    code = (
        "import json, os, subprocess, sys, time; from pathlib import Path; "
        f"worker = subprocess.Popen([sys.executable, '-c', {worker!r}]); "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid(), worker.pid])); "
    )
    if not parent_exits:
        code += "time.sleep(5)"

    with pytest.raises(TimeoutError, match="timed out"):
        execute_subprocess_async([sys.executable, "-c", code], timeout=1, quiet=True, echo=False)

    assert_processes_stopped(child_processes)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_subprocess_cancellation_stops_child(child_processes):
    code = (
        "import json, os, time; from pathlib import Path; "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid()])); time.sleep(5)"
    )

    async def cancel_after_start():
        task = asyncio.create_task(_stream_subprocess([sys.executable, "-c", code], quiet=True))
        try:
            for _ in range(300):
                if child_processes.exists():
                    break
                await asyncio.sleep(0.01)
            assert child_processes.exists(), "Child did not start"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert_processes_stopped(child_processes)

    asyncio.run(cancel_after_start())


def test_subprocess_preserves_environment_stdin_and_echo(tmp_path, capsys):
    input_file = tmp_path / "input.txt"
    input_file.write_text("input marker")
    env = dict(os.environ, ACCELERATE_TIMEOUT_TEST="environment marker")
    with input_file.open() as stdin:
        result = execute_subprocess_async(
            [
                sys.executable,
                "-c",
                "import os, sys; print(os.environ['ACCELERATE_TIMEOUT_TEST']); "
                "print(sys.stdin.read()); print('diagnostic', file=sys.stderr)",
            ],
            env=env,
            stdin=stdin,
            timeout=5,
        )

    assert result.stdout == ["environment marker", "input marker"]
    assert result.stderr == ["diagnostic"]
    captured = capsys.readouterr()
    assert "Running:" in captured.out
    assert "stdout: environment marker" in captured.out
    assert "stderr: diagnostic" in captured.err


def test_subprocess_zero_timeout_reaps_child(monkeypatch):
    processes = []
    create_subprocess = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await create_subprocess(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)
    with pytest.raises(TimeoutError, match="timed out"):
        execute_subprocess_async(
            [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0, quiet=True, echo=False
        )

    assert len(processes) == 1
    assert processes[0].returncode is not None


def test_subprocess_missing_command_preserves_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        execute_subprocess_async([tmp_path / "missing-command"], timeout=1, quiet=True, echo=False)


def test_subprocess_timeout_without_process_groups(child_processes, monkeypatch):
    # Exercise the direct-child fallback with a real process, without pretending
    # this is native Windows event-loop or descendant-cleanup coverage.
    monkeypatch.setattr(testing, "os", SimpleNamespace(name="nt"))
    code = (
        "import json, os, time; from pathlib import Path; "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid()])); time.sleep(5)"
    )
    with pytest.raises(TimeoutError, match="timed out"):
        execute_subprocess_async([sys.executable, "-c", code], timeout=1, quiet=True, echo=False)

    assert_processes_stopped(child_processes)


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGINT behavior")
def test_subprocess_keyboard_interrupt_stops_child(child_processes):
    child = (
        "import json, os, time; from pathlib import Path; "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid()])); time.sleep(5)"
    )
    supervisor_code = (
        "import sys\n"
        "from accelerate.test_utils.testing import execute_subprocess_async\n"
        "try:\n"
        f"    execute_subprocess_async([sys.executable, '-c', {child!r}], timeout=None, quiet=True, echo=False)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted')\n"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", supervisor_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        for _ in range(500):
            if child_processes.exists():
                break
            time.sleep(0.01)
        assert child_processes.exists(), "Supervised child did not start"
        supervisor.send_signal(signal.SIGINT)
        stdout, stderr = supervisor.communicate(timeout=5)
        assert supervisor.returncode == 0, stderr.decode()
        assert b"interrupted" in stdout
        assert_processes_stopped(child_processes)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.communicate(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX child cleanup")
def test_subprocess_reader_failure_stops_child(child_processes):
    code = (
        "import json, os, time; from pathlib import Path; "
        f"Path({str(child_processes)!r}).write_text(json.dumps([os.getpid()])); "
        "os.write(1, bytes([255, 10])); time.sleep(5)"
    )

    with pytest.raises(UnicodeDecodeError):
        execute_subprocess_async([sys.executable, "-c", code], timeout=2, quiet=True, echo=False)

    assert_processes_stopped(child_processes)
