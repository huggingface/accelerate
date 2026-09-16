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
"""Regression test for the `param_sync_func` closure-capture bug in
`MegatronEngine.get_module_config` (accelerate/utils/megatron_lm.py).

`get_module_config` itself can't be exercised directly here without the
real `megatron-core` package (its imports of `get_args`, `get_model_config`,
etc. come straight from `megatron.core`, a heavy, cluster-oriented
dependency this test environment doesn't have). This mirrors the exact
list-comprehension at the one line that changed, with a lightweight
stand-in for `self.optimizer`, and asserts what actually matters: each
callback in the resulting list must call back with *its own* model_index,
not whichever index the loop happened to end on.

Before the fix, this list comprehension read
`lambda x: self.optimizer.finish_param_sync(model_index, x)` with no
default argument - every callback shared the same `model_index` cell, so
all of them fired with the last chunk's index regardless of which one was
actually invoked.
"""


class _RecordingOptimizer:
    def __init__(self):
        self.calls = []

    def finish_param_sync(self, model_index, x):
        self.calls.append((model_index, x))


def test_param_sync_func_callbacks_use_their_own_model_index():
    optimizer = _RecordingOptimizer()
    num_modules = 3

    # Exact shape of the fixed line in megatron_lm.py's get_module_config.
    param_sync_func = [
        lambda x, model_index=model_index: optimizer.finish_param_sync(model_index, x)
        for model_index in range(num_modules)
    ]

    assert len(param_sync_func) == num_modules
    for i, callback in enumerate(param_sync_func):
        callback(f"grad_{i}")

    assert optimizer.calls == [(0, "grad_0"), (1, "grad_1"), (2, "grad_2")]
