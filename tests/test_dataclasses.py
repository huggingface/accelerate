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

import unittest

from accelerate.utils import ParallelismConfig


class TestParallelismConfig(unittest.TestCase):

    def test_get_mesh_fsdp(self):
        """Test pure FSDP (dp_shard only)."""
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=8, tp_size=1)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (8,))
        self.assertEqual(mesh_dim_names, ("dp_shard",))
        
    def test_get_mesh_ddp(self):
        """Test pure DDP (dp_replicate only)."""
        config = ParallelismConfig(dp_replicate_size=8, dp_shard_size=1, tp_size=1)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (8,))
        self.assertEqual(mesh_dim_names, ("dp_replicate",))
        
    def test_get_mesh_hsdp(self):
        """Test pure HSDP (both dp_replicate and dp_shard)."""
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=4, tp_size=1)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (2, 4))
        self.assertEqual(mesh_dim_names, ("dp_replicate", "dp_shard"))
        
    def test_get_mesh_fsdp_tp(self):
        """Test FSDP + TP combination."""
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=4, tp_size=2)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (4, 2))
        self.assertEqual(mesh_dim_names, ("dp_shard", "tp"))
        
    def test_get_mesh_ddp_tp(self):
        """Test DDP + TP combination."""
        config = ParallelismConfig(dp_replicate_size=4, dp_shard_size=1, tp_size=2)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (4, 2))
        self.assertEqual(mesh_dim_names, ("dp_replicate", "tp"))
        
    def test_get_mesh_hsdp_tp(self):
        """Test 3D parallelism: HSDP + TP combination."""
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=2)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (2, 2, 2))
        self.assertEqual(mesh_dim_names, ("dp_replicate", "dp_shard", "tp"))
        
    def test_get_mesh_tp(self):
        """Test only TP."""
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=8)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, (8,))
        self.assertEqual(mesh_dim_names, ("tp",))
        
    def test_get_mesh_no_parallelism(self):
        """Test no parallelism."""
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1)
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, ())
        self.assertEqual(mesh_dim_names, ())