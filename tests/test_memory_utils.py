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

import os
import sys
import tempfile
import unittest
from unittest import mock

import psutil
from torch import nn

from accelerate.test_utils import (
    memory_allocated_func,
    require_non_cpu,
    require_non_torch_xla,
    torch_device,
)
from accelerate.utils import memory as memory_utils
from accelerate.utils.memory import find_executable_batch_size, get_cgroup_available_memory, release_memory
from accelerate.utils.modeling import get_max_memory


def raise_fake_out_of_memory():
    raise RuntimeError(f"{torch_device.upper()} out of memory.")


class ModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class BigModelForTest(ModelForTest):
    def __init__(self):
        super().__init__()
        self.linear3 = nn.Linear(5, 1000)

    def forward(self, x):
        return self.linear3(super().forward(x))


class MemoryTest(unittest.TestCase):
    def test_memory_implicit(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size != 8:
                raise_fake_out_of_memory()

        mock_training_loop_function()
        assert batch_sizes == [
            128,
            115,
            103,
            92,
            82,
            73,
            65,
            58,
            52,
            46,
            41,
            36,
            32,
            28,
            25,
            22,
            19,
            17,
            15,
            13,
            11,
            9,
            8,
        ]

    def test_memory_explicit(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size, arg1):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size != 8:
                raise_fake_out_of_memory()
            return batch_size, arg1

        bs, arg1 = mock_training_loop_function("hello")
        assert batch_sizes == [
            128,
            115,
            103,
            92,
            82,
            73,
            65,
            58,
            52,
            46,
            41,
            36,
            32,
            28,
            25,
            22,
            19,
            17,
            15,
            13,
            11,
            9,
            8,
        ]
        assert [bs, arg1] == [8, "hello"]

    def test_start_zero(self):
        @find_executable_batch_size(starting_batch_size=0)
        def mock_training_loop_function(batch_size):
            pass

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            assert "No executable batch size found, reached zero." in cm.exception.args[0]

    def test_approach_zero(self):
        @find_executable_batch_size(starting_batch_size=16)
        def mock_training_loop_function(batch_size):
            if batch_size > 0:
                raise_fake_out_of_memory()
            pass

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            assert "No executable batch size found, reached zero." in cm.exception.args[0]

    def test_verbose_guard(self):
        @find_executable_batch_size(starting_batch_size=128)
        def mock_training_loop_function(batch_size, arg1, arg2):
            if batch_size != 8:
                raise raise_fake_out_of_memory()

        with self.assertRaises(TypeError) as cm:
            mock_training_loop_function(128, "hello", "world")
            assert "Batch size was passed into `f`" in cm.exception.args[0]
            assert "`f(arg1='hello', arg2='world')" in cm.exception.args[0]

    def test_any_other_error(self):
        @find_executable_batch_size(starting_batch_size=16)
        def mock_training_loop_function(batch_size):
            raise ValueError("Oops, we had an error!")

        with self.assertRaises(ValueError) as cm:
            mock_training_loop_function()
            assert "Oops, we had an error!" in cm.exception.args[0]

    @require_non_cpu
    @require_non_torch_xla
    def test_release_memory(self):
        starting_memory = memory_allocated_func()

        if torch_device.startswith("hpu"):
            # hpu has a minimum memory allocation that cannot be released,
            # we need to surpass it by using a bigger model (>5767296 bytes)
            model = BigModelForTest()
        else:
            model = ModelForTest()

        model.to(torch_device)
        assert memory_allocated_func() > starting_memory
        model = release_memory(model)
        assert memory_allocated_func() == starting_memory


MiB = 1 << 20
GiB = 1 << 30


class CgroupMemoryTest(unittest.TestCase):
    """`get_cgroup_available_memory` must honour every cgroup limit the kernel enforces, from the process's cgroup up
    to the root, resolving the cgroup directories through `/proc/self/mountinfo` like the kernel does."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "cgroup")
        self.proc_self_cgroup = os.path.join(self.tmp.name, "proc_self_cgroup")
        # Not created unless a test writes it, so the real /proc/self/mountinfo is never read and the fallback is
        # `_CGROUP_ROOT`, which points into the temporary tree.
        self.proc_self_mountinfo = os.path.join(self.tmp.name, "proc_self_mountinfo")
        for patcher in (
            mock.patch.object(memory_utils, "_CGROUP_ROOT", self.root),
            mock.patch.object(memory_utils, "_PROC_SELF_CGROUP", self.proc_self_cgroup),
            mock.patch.object(memory_utils, "_PROC_SELF_MOUNTINFO", self.proc_self_mountinfo),
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.dict(os.environ, {"ACCELERATE_RESPECT_CGROUP_MEMORY": "1"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write_tree(self, proc_self_cgroup, files, mountinfo=None):
        """Writes the fake `/proc/self/cgroup`, the fake `/proc/self/mountinfo` (a list of (mount root, mount point,
        fs type, super options)) and the cgroup files. File paths are relative to the fake cgroup root unless they
        are absolute, which the mount tests use to place files under their own mount points."""
        with open(self.proc_self_cgroup, "w") as f:
            f.write(proc_self_cgroup + "\n")
        if mountinfo is not None:
            with open(self.proc_self_mountinfo, "w") as f:
                for i, (mount_root, mount_point, fs_type, super_options) in enumerate(mountinfo):
                    f.write(f"{40 + i} 25 0:30 {mount_root} {mount_point} rw - {fs_type} cgroup {super_options}\n")
        for rel_path, content in files.items():
            path = os.path.join(self.root, rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content + "\n")

    def mount(self, name):
        return os.path.join(self.tmp.name, "mnt", name)

    def test_limit_on_parent_cgroup(self):
        # Kubernetes sets the pod limit on the parent cgroup; the container's own memory.max is `max`.
        self.write_tree(
            "0::/pod/ctr",
            {
                "memory.max": "max",
                "pod/memory.max": str(8 * GiB),
                "pod/memory.current": str(3 * GiB),
                "pod/memory.stat": f"anon {2 * GiB}\ninactive_file {GiB}",
                "pod/ctr/memory.max": "max",
                "pod/ctr/memory.current": str(GiB),
            },
        )
        # 8 GiB limit - (3 GiB usage - 1 GiB reclaimable page cache)
        assert get_cgroup_available_memory() == 6 * GiB

    def test_smallest_limit_wins(self):
        self.write_tree(
            "0::/pod/ctr",
            {
                "memory.max": str(4 * GiB),
                "memory.current": str(GiB),
                "pod/memory.max": str(16 * GiB),
                "pod/memory.current": str(GiB),
                "pod/ctr/memory.max": str(32 * GiB),
            },
        )
        assert get_cgroup_available_memory() == 3 * GiB

    def test_cgroup_v1_sentinel_is_no_limit(self):
        self.write_tree(
            "12:memory:/pod\n3:cpu,cpuacct:/pod",
            {
                "memory/memory.limit_in_bytes": "9223372036854771712",
                "memory/pod/memory.limit_in_bytes": str(2 * GiB),
                "memory/pod/memory.usage_in_bytes": str(GiB),
                "memory/pod/memory.stat": "total_inactive_file 0",
            },
        )
        assert get_cgroup_available_memory() == GiB

    def test_sibling_usage_counts_against_parent_limit(self):
        # The container's own limit is the smallest, but the parent has less room left because a sibling uses it.
        self.write_tree(
            "0::/pod/ctr",
            {
                "memory.max": "max",
                "pod/memory.max": str(2 * GiB),
                "pod/memory.current": str(7 * GiB // 4),
                "pod/ctr/memory.max": str(GiB),
                "pod/ctr/memory.current": "0",
            },
        )
        assert get_cgroup_available_memory() == GiB // 4

    def test_subtree_alias_does_not_hide_parent_limit(self):
        # The full hierarchy and a bind mount of the container's subtree both contain the process. The full view
        # must win whatever the mount order, because the parent limit still applies.
        main, ctr = self.mount("main"), self.mount("ctr")
        mounts = [("/", main, "cgroup2", "rw"), ("/pod/ctr", ctr, "cgroup2", "rw")]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.write_tree(
                    "0::/pod/ctr",
                    {
                        f"{main}/memory.max": "max",
                        f"{main}/pod/memory.max": str(512 * MiB),
                        f"{main}/pod/memory.current": str(256 * MiB),
                        f"{main}/pod/ctr/memory.max": str(GiB),
                        f"{main}/pod/ctr/memory.current": "0",
                        f"{ctr}/memory.max": str(GiB),
                        f"{ctr}/memory.current": "0",
                    },
                    mountinfo=mounts[::-1] if reverse else mounts,
                )
                assert get_cgroup_available_memory() == 256 * MiB

    def test_cgroup_v1_mount_root_is_stripped(self):
        # Docker with the host cgroup namespace: /proc/self/cgroup names the cgroup relative to the hierarchy root,
        # while the mount exposes only the container's subtree.
        memory = self.mount("memory")
        self.write_tree(
            "12:memory:/docker/abc/pod/app\n3:cpu,cpuacct:/docker/abc/pod/app",
            {
                f"{memory}/memory.limit_in_bytes": str(GiB),
                f"{memory}/memory.usage_in_bytes": str(128 * MiB),
                f"{memory}/pod/app/memory.limit_in_bytes": str(256 * MiB),
                f"{memory}/pod/app/memory.usage_in_bytes": str(64 * MiB),
            },
            mountinfo=[("/docker/abc", memory, "cgroup", "rw,memory")],
        )
        assert get_cgroup_available_memory() == 192 * MiB

    def test_mount_point_with_escaped_space(self):
        # mountinfo escapes a space in a path as \040.
        mount_point = self.mount("my cg")
        self.write_tree(
            "0::/job",
            {f"{mount_point}/job/memory.max": str(GiB), f"{mount_point}/job/memory.current": str(256 * MiB)},
            mountinfo=[("/", mount_point.replace(" ", "\\040"), "cgroup2", "rw")],
        )
        assert get_cgroup_available_memory() == 768 * MiB

    def test_no_limit_returns_none(self):
        self.write_tree("0::/", {"memory.max": "max", "memory.current": str(GiB)})
        assert get_cgroup_available_memory() is None

    def test_default_and_override(self):
        self.write_tree("0::/", {"memory.max": str(2 * GiB), "memory.current": str(GiB)})
        cases = [(False, None, None), (True, None, GiB), (False, "1", GiB), (True, "0", None)]
        for default, override, expected in cases:
            env = {} if override is None else {"ACCELERATE_RESPECT_CGROUP_MEMORY": override}
            with (
                self.subTest(default=default, override=override),
                mock.patch.object(memory_utils, "RESPECT_CGROUP_MEMORY_BY_DEFAULT", default),
                mock.patch.dict(os.environ, env, clear=False),
            ):
                if override is None:
                    os.environ.pop("ACCELERATE_RESPECT_CGROUP_MEMORY", None)
                assert get_cgroup_available_memory() == expected

    @mock.patch("accelerate.utils.modeling.torch.cuda.is_available", return_value=False)
    @mock.patch("accelerate.utils.modeling.torch.accelerator.is_available", return_value=False)
    def test_get_max_memory_caps_cpu_at_cgroup_limit(self, *_):
        self.write_tree("0::/", {"memory.max": str(GiB), "memory.current": "0"})
        assert get_max_memory()["cpu"] == min(GiB, psutil.virtual_memory().available)
