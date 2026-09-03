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
import unittest
from dataclasses import dataclass, field

import pytest

from accelerate.utils.error_reporting import (
    SUMMARY_ENV_VAR,
    classify_rank_failures,
    format_failure_summary,
    summarize_child_failure,
)


@dataclass
class FakeFailure:
    """Duck-typed stand-in for torch.distributed.elastic's ProcessFailure."""

    local_rank: int
    pid: int
    exitcode: int
    timestamp: int
    error_file: str = "<N/A>"
    message: str = ""  # required by ChildFailedError.format_msg() in the Tier-2a tests
    _signal: str | None = field(default=None)

    def signal_name(self):
        return self._signal or "<N/A>"

    def timestamp_isoformat(self):
        # Also required by ChildFailedError.format_msg(); a fixed stamp keeps tests deterministic.
        return "2026-07-28_00:00:00"


def _sigabrt(rank, ts=100):
    return FakeFailure(local_rank=rank, pid=1000 + rank, exitcode=-6, timestamp=ts, _signal="SIGABRT")


def _sigterm(rank, ts=105):
    return FakeFailure(local_rank=rank, pid=1000 + rank, exitcode=-15, timestamp=ts, _signal="SIGTERM")


def _pyexit(rank, code=1, ts=100):
    return FakeFailure(local_rank=rank, pid=1000 + rank, exitcode=code, timestamp=ts)


class ClassifyRankFailuresTester(unittest.TestCase):
    def test_watchdog_style_abort_is_bucketed_and_mapped(self):
        info = classify_rank_failures({1: _sigabrt(1, ts=100), 0: _sigterm(0, ts=105), 2: _sigterm(2, ts=106)})
        assert info["first_rank"] == 1  # earliest timestamp, not lowest rank
        assert info["aborted"] == [1]
        assert info["torn_down"] == [0, 2]
        assert info["raised"] == []
        assert info["exitcode"] == 134  # 128 + SIGABRT(6)

    def test_ordinary_python_exit_is_not_mislabelled_as_watchdog(self):
        info = classify_rank_failures({0: _pyexit(0)})
        assert info["aborted"] == []
        assert info["raised"] == [0]
        assert info["exitcode"] == 1

    def test_empty_failures_never_raise(self):
        info = classify_rank_failures({})
        assert info["first"] is None
        assert info["exitcode"] == 1

    def test_signal_name_shape_drift_is_tolerated(self):
        class WeirdFailure:
            local_rank, pid, exitcode, timestamp, error_file = 0, 1, -9, 100, "<N/A>"
            signal_name = "SIGKILL"  # attribute instead of method

        info = classify_rank_failures({0: WeirdFailure()})
        assert info["torn_down"] == [0]
        assert info["exitcode"] == 137


class FormatFailureSummaryTester(unittest.TestCase):
    def test_watchdog_abort_report_content(self):
        out = format_failure_summary(
            {1: _sigabrt(1, ts=100), 0: _sigterm(0, ts=105), 2: _sigterm(2, ts=106)}, entrypoint="train.py"
        )
        assert out.count("First observed failure") == 1
        assert "HEURISTIC" in out  # honesty caveat is load-bearing, not decoration
        assert "NCCL watchdog" in out
        assert "collateral, not causes" in out
        assert "--local_ranks_filter 0" in out  # re-run recipe present
        assert "`train.py` failed on 3 rank(s)" in out

    def test_python_failure_report_has_no_watchdog_claim(self):
        out = format_failure_summary({0: _pyexit(0)}, entrypoint="train.py")
        assert "NCCL watchdog" not in out
        assert "record" in out  # points at @record for tracebacks

    def test_missing_error_files_are_reported_honestly(self):
        out = format_failure_summary({1: _sigabrt(1)}, entrypoint="train.py")
        assert "none -- worker entrypoints were not decorated" in out


class SummarizeChildFailureTester(unittest.TestCase):
    class FakeChildFailedError(Exception):
        def __init__(self, failures):
            self.failures = failures

    def test_env_kill_switch_disables_summary(self):
        error = self.FakeChildFailedError({1: _sigabrt(1)})
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(SUMMARY_ENV_VAR, "0")
            assert summarize_child_failure(error) is None

    def test_error_without_failures_returns_none(self):
        assert summarize_child_failure(RuntimeError("not a child failure")) is None

    def test_formatter_bug_degrades_to_none_never_raises(self):
        class Poison:
            def __getattr__(self, name):
                raise ValueError("poisoned attribute access")

        error = self.FakeChildFailedError({0: Poison()})
        assert summarize_child_failure(error) is None  # zero-blast-radius gate


class LauncherClauseTester(unittest.TestCase):
    """Tier 2a: exercise the real `except ChildFailedError` clause in multi_gpu_launcher
    by monkeypatching `distrib_run.run` — reachable on a zero-GPU runner. (A `--cpu`
    subprocess test would dispatch to `simple_launcher` and never reach this clause.)
    """

    def _launch_args(self, extra=None):
        from accelerate.commands.launch import launch_command_parser

        argv = [
            "--multi_gpu",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            "train.py",
        ]
        if extra:
            argv = extra + argv
        args = launch_command_parser().parse_args(argv)
        args.debug = False
        return args

    def _synthesized_error(self):
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        return ChildFailedError("train.py", {1: _sigabrt(1, ts=100), 0: _sigterm(0, ts=105)})

    def test_clause_prints_one_summary_and_reraises_by_default(self):
        import torch.distributed.run as distrib_run
        from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

        from accelerate.commands import launch as launch_module

        args = self._launch_args()
        error = self._synthesized_error()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(distrib_run, "run", lambda a: (_ for _ in ()).throw(error))
            with pytest.raises(ChildFailedError), _capture_stderr() as captured:
                launch_module.multi_gpu_launcher(args)
        assert captured["value"].count("First observed failure") == 1

    def test_quiet_exits_with_signal_mapped_code(self):
        import torch.distributed.run as distrib_run

        from accelerate.commands import launch as launch_module

        args = self._launch_args(extra=["--quiet"])
        error = self._synthesized_error()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(distrib_run, "run", lambda a: (_ for _ in ()).throw(error))
            with pytest.raises(SystemExit) as exc_info, _capture_stderr() as captured:
                launch_module.multi_gpu_launcher(args)
        assert exc_info.value.code == 134
        assert captured["value"].count("First observed failure") == 1


class _capture_stderr:
    def __init__(self):
        self._buf = None

    def __enter__(self):
        import io
        import sys

        self._old = sys.stderr
        self._buf = io.StringIO()
        sys.stderr = self._buf
        self.result = {"value": ""}
        return self.result

    def __exit__(self, *exc):
        import sys

        self.result["value"] = self._buf.getvalue()
        sys.stderr = self._old
        return False
