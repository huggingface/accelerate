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
"""
Test file to ensure that in general certain situational setups for notebooks work.
"""

import os
import time
from multiprocessing import Queue

from pytest import mark, raises
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available


def basic_function():
    # Just prints the PartialState
    print(f"PartialState:\n{PartialState()}")


def tough_nut_function(queue: Queue):
    if queue.empty():
        return
    trial = queue.get()
    if trial > 0:
        queue.put(trial - 1)
        raise RuntimeError("The nut hasn't cracked yet! Try again.")

    print(f"PartialState:\n{PartialState()}")


def bipolar_sleep_function(sleep_sec: int):
    state = PartialState()
    if state.process_index % 2 == 0:
        raise RuntimeError("I'm an even process. I don't like to sleep.")
    else:
        time.sleep(sleep_sec)


NUM_PROCESSES = int(os.environ.get("ACCELERATE_NUM_PROCESSES", 1))


def test_can_initialize():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


@mark.skipif(NUM_PROCESSES < 2, reason="Need at least 2 processes to test static rendezvous backends")
def test_static_rdzv_backend():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES, rdzv_backend="static")


@mark.skipif(NUM_PROCESSES < 2, reason="Need at least 2 processes to test c10d rendezvous backends")
def test_c10d_rdzv_backend():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES, rdzv_backend="c10d")


@mark.skipif(NUM_PROCESSES < 2, reason="Need at least 2 processes to test fault tolerance")
def test_fault_tolerant(max_restarts: int = 3):
    queue = Queue()
    queue.put(max_restarts)
    notebook_launcher(tough_nut_function, (queue,), num_processes=NUM_PROCESSES, max_restarts=max_restarts)


@mark.skipif(NUM_PROCESSES < 2, reason="Need at least 2 processes to test monitoring")
def test_monitoring(monitor_interval: float = 0.01, sleep_sec: int = 100):
    start_time = time.time()
    with raises(ChildFailedError, match="I'm an even process. I don't like to sleep."):
        notebook_launcher(
            bipolar_sleep_function,
            (sleep_sec,),
            num_processes=NUM_PROCESSES,
            monitor_interval=monitor_interval,
        )
    assert time.time() - start_time < sleep_sec, "Monitoring did not stop the process in time."


@require_bnb
def test_problematic_imports():
    with raises(RuntimeError, match="Please keep these imports"):
        import bitsandbytes as bnb  # noqa: F401

        notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


def main():
    print("Test basic notebook can be ran")
    test_can_initialize()
    print("Test static rendezvous backend")
    test_static_rdzv_backend()
    print("Test c10d rendezvous backend")
    test_c10d_rdzv_backend()
    print("Test fault tolerant")
    test_fault_tolerant()
    print("Test monitoring")
    test_monitoring()
    if is_bnb_available():
        print("Test problematic imports (bnb)")
        test_problematic_imports()
    if NUM_PROCESSES > 1:
        PartialState().destroy_process_group()


if __name__ == "__main__":
    main()
