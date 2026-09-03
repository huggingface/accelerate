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
"""
Pure helpers that turn ``torch.distributed.elastic`` per-rank failure data into a
single actionable console diagnostic.

Deliberately narrow by design:

* Nothing in this module issues a collective, touches a process group, or blocks.
* Nothing in this module imports ``torch`` — it reads already-materialized
  ``ProcessFailure`` metadata duck-typed via ``getattr``, so it is unit-testable
  on a CPU-only runner with plain fakes.
* Formatting failures must never mask the original crash: the public entrypoint
  ``summarize_child_failure`` returns ``None`` on any internal error, and callers
  are expected to fall back to re-raising the original exception unchanged.
"""

from __future__ import annotations

import os
from typing import Any


__all__ = [
    "SUMMARY_ENV_VAR",
    "classify_rank_failures",
    "format_failure_summary",
    "summarize_child_failure",
]

# Setting ACCELERATE_DISTRIBUTED_ERROR_SUMMARY=0 disables the diagnostic block
# entirely, restoring byte-identical previous behavior.
SUMMARY_ENV_VAR = "ACCELERATE_DISTRIBUTED_ERROR_SUMMARY"

# SIGABRT is the signature of PyTorch's NCCL watchdog / heartbeat monitor taking
# the process down (std::terminate on the watchdog thread). The other entries are
# hard faults that likewise never surface as Python exceptions.
_ABORT_SIGNALS = frozenset({"SIGABRT", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL"})
# Ranks killed by the elastic agent after another rank failed. Their tracebacks
# are collateral of the group teardown, not evidence about the root cause.
_TEARDOWN_SIGNALS = frozenset({"SIGTERM", "SIGINT", "SIGKILL"})

_WATCHDOG_EXPLANATION = (
    "A rank exiting on SIGABRT is the signature of PyTorch's NCCL watchdog or heartbeat "
    "monitor tearing the process down after a collective exceeded its timeout. This is by "
    "design and is not catchable in Python: the abort is raised on a C++ watchdog thread, "
    "not on the thread that issued the collective."
)
_VICTIM_WARNING = (
    "The rank that reports a timeout is frequently a VICTIM waiting on a peer. The "
    "root-cause rank is often the one that produced no output at all (it hung), so treat "
    "'first observed failure' as a heuristic, not a verdict."
)


def _signal_name(failure: Any) -> str | None:
    """``ProcessFailure.signal_name`` is a method upstream; tolerate shape drift."""
    name = getattr(failure, "signal_name", None)
    try:
        name = name() if callable(name) else name
    except Exception:
        return None
    # Upstream uses "<N/A>" for non-signal exits; normalize anything non-signal-like to None.
    if not name or not str(name).startswith("SIG"):
        return None
    return str(name)


def classify_rank_failures(failures: dict[int, Any]) -> dict[str, Any]:
    """
    Partition ``ChildFailedError.failures`` into diagnostic buckets.

    Returns a dict with ``first_rank``, ``first`` (the ``ProcessFailure`` with the
    earliest timestamp — torchelastic's own, explicitly heuristic, root-cause rule),
    ``aborted`` / ``torn_down`` / ``raised`` rank lists, and ``exitcode`` mapped to
    the shell convention (a signal death of ``-N`` becomes ``128 + N``).
    """
    if not failures:
        return {
            "first_rank": None,
            "first": None,
            "aborted": [],
            "torn_down": [],
            "raised": [],
            "exitcode": 1,
        }

    first_rank = min(failures, key=lambda rank: getattr(failures[rank], "timestamp", 0))
    aborted, torn_down, raised = [], [], []
    for rank in sorted(failures):
        name = _signal_name(failures[rank])
        if name in _ABORT_SIGNALS:
            aborted.append(rank)
        elif name in _TEARDOWN_SIGNALS:
            torn_down.append(rank)
        else:
            raised.append(rank)

    exitcode = getattr(failures[first_rank], "exitcode", 1) or 1
    return {
        "first_rank": first_rank,
        "first": failures[first_rank],
        "aborted": aborted,
        "torn_down": torn_down,
        "raised": raised,
        # Shell exit codes are unsigned: map a signal death (-N) to 128 + N.
        "exitcode": (128 - exitcode) if exitcode < 0 else exitcode,
    }


def format_failure_summary(
    failures: dict[int, Any],
    *,
    entrypoint: str = "training script",
    log_dir: str | None = None,
    tee: str | None = None,
    width: int = 78,
) -> str:
    """Render one bordered, self-contained diagnostic block. Pure; never raises by contract of its caller."""
    info = classify_rank_failures(failures)
    first, first_rank = info["first"], info["first_rank"]
    border, rule = "=" * width, "-" * width
    out = [border, f"accelerate launch: `{entrypoint}` failed on {len(failures)} rank(s)", rule]
    if first is None:
        return "\n".join(out + ["No per-rank failure detail was reported by the launcher.", border])

    sig = _signal_name(first)
    error_file = getattr(first, "error_file", None)
    out += [
        "First observed failure (earliest timestamp -- HEURISTIC, not a verdict):",
        f"  rank      : {first_rank} (local_rank {getattr(first, 'local_rank', '?')})",
        f"  exitcode  : {getattr(first, 'exitcode', '?')}"
        + (f" ({sig})" if sig else "")
        + f", pid {getattr(first, 'pid', '?')}",
        f"  error_file: {error_file if error_file and error_file != '<N/A>' else '<none written -- see @record note below>'}",
        "",
    ]
    if sig in _ABORT_SIGNALS:
        out += [f"  {_WATCHDOG_EXPLANATION}", ""]
    out += [f"  {_VICTIM_WARNING}", "", "Rank breakdown:"]
    for label, ranks in (
        ("aborted (watchdog / hard fault)", info["aborted"]),
        ("torn down by the launcher", info["torn_down"]),
        ("raised a Python exception / exited", info["raised"]),
    ):
        out.append(f"  {label:<36}: {len(ranks)}" + (f"  -> ranks {ranks}" if ranks else ""))
    if info["torn_down"]:
        out += [
            "",
            "  Ranks torn down by SIGTERM/SIGKILL were killed by the elastic agent AFTER",
            "  the first failure. Their tracebacks are collateral, not causes.",
        ]

    out += ["", "Preserved per-rank detail:"]
    wrote_any = False
    for rank in sorted(failures):
        path = getattr(failures[rank], "error_file", None)
        if path and path != "<N/A>" and os.path.exists(path):
            out.append(f"  rank {rank}: {path}")
            wrote_any = True
    if not wrote_any:
        out += [
            "  none -- worker entrypoints were not decorated with",
            "  `torch.distributed.elastic.multiprocessing.errors.record`, so no error",
            "  files were written (exit codes and signals only).",
        ]
    if not log_dir or tee in (None, "", "0"):
        out += [
            "  Per-rank stdout/stderr was NOT captured to disk for this run.",
            "  Re-run with:  accelerate launch --tee 3 --log_dir ./accelerate_logs ...",
        ]
    else:
        out.append(f"  Per-rank stdout/stderr: {log_dir}")

    out += [
        "",
        "Suggested next steps:",
        "  1. Re-run with per-rank capture and a quiet console:",
        "       accelerate launch --tee 3 --log_dir ./accelerate_logs \\",
        "                         --local_ranks_filter 0 ...",
        "  2. Enable Flight Recorder to see which collective each rank was in:",
        "       TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1",
        "  3. Look first at the rank that produced NO output.",
        border,
    ]
    return "\n".join(out)


def summarize_child_failure(error: Any, **kwargs: Any) -> str | None:
    """
    Best-effort wrapper: ``ChildFailedError`` -> one diagnostic string.

    Returns ``None`` (meaning: caller re-raises exactly as before) when the summary
    is disabled via the environment, when the error carries no per-rank failures, or
    when formatting itself fails for any reason. A bug in the diagnostics must never
    hide the actual crash.
    """
    if os.environ.get(SUMMARY_ENV_VAR, "1").lower() in ("0", "false", "no", "off"):
        return None
    failures = getattr(error, "failures", None)
    if not failures:
        return None
    try:
        return format_failure_summary(failures, **kwargs)
    except Exception:
        return None
