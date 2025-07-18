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

from accelerate.utils import ParallelismConfig
from accelerate.test_utils.testing import AccelerateTestCase


class TestParallelismConfig(AccelerateTestCase):

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

    def test_get_mesh_ordering(self):
        """Test that mesh dimensions are properly ordered."""
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=4, tp_size=2)
        mesh_shape, mesh_dim_names = config.get_mesh()
        
        self.assertEqual(mesh_dim_names, ("dp_replicate", "dp_shard", "tp"))
        self.assertEqual(mesh_shape, (2, 4, 2))

    def test_validate_accelerator_total_size_mismatch(self):
        """Test validate_accelerator with total_size mismatch."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=2)  # total_size = 8
        config._is_fully_initialized = True
        
        accelerator = Accelerator() 
        
        with self.assertRaises(ValueError) as cm:
            config.validate_accelerator(accelerator)
        self.assertIn("requires 8 processes, but the current Accelerator is set to use 1 processes", str(cm.exception))

    def test_validate_accelerator_not_initialized(self):
        """Test validate_accelerator with uninitialized config."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1) 
        accelerator = Accelerator()
        
        with self.assertRaises(ValueError) as cm:
            config.validate_accelerator(accelerator)
        self.assertIn("ParallelismConfig is not fully initialized", str(cm.exception))

    def test_validate_accelerator_success(self):
        """Test validate_accelerator with matching config."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1) 
        config._is_fully_initialized = True
        
        accelerator = Accelerator()
        config.validate_accelerator(accelerator)

    def test_validate_device_mesh_size_mismatch(self):
        """Test validate_device_mesh with size mismatch."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1)  
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 8  # Wrong size
        mock_mesh.mesh_dim_names = ("dp_replicate", "dp_shard")
        
        with self.assertRaises(ValueError) as cm:
            config.validate_device_mesh(mock_mesh)
        self.assertIn("Device mesh size 8 does not match the total size of the parallelism config 4", str(cm.exception))

    def test_validate_device_mesh_invalid_dims(self):
        """Test validate_device_mesh with invalid dimension names."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1)
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 4
        mock_mesh.mesh_dim_names = ("dp_replicate", "invalid_dim")
        
        with self.assertRaises(ValueError) as cm:
            config.validate_device_mesh(mock_mesh)
        self.assertIn("contain invalid dimensions", str(cm.exception))

    def test_validate_device_mesh_dimension_mismatch(self):
        """Test validate_device_mesh with dimension mismatch."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1)  
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 4
        mock_mesh.mesh_dim_names = ("dp_replicate", "tp") 
        
        with self.assertRaises(ValueError) as cm:
            config.validate_device_mesh(mock_mesh)
        self.assertIn("do not match the expected dimensions", str(cm.exception))

    def test_validate_device_mesh_dimension_order_mismatch(self):
        """Test validate_device_mesh with dimension order mismatch."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1)  
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 4
        mock_mesh.mesh_dim_names = ("dp_shard", "dp_replicate")  
        mock_mesh.shape = (2, 2)
        
        with self.assertRaises(ValueError) as cm:
            config.validate_device_mesh(mock_mesh)
        self.assertIn("Device mesh dimension order mismatch", str(cm.exception))

    def test_validate_device_mesh_shape_mismatch(self):
        """Test validate_device_mesh with shape mismatch."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1) 
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 4
        mock_mesh.mesh_dim_names = ("dp_replicate", "dp_shard")
        mock_mesh.shape = (4, 1) 
        
        with self.assertRaises(ValueError) as cm:
            config.validate_device_mesh(mock_mesh)
        self.assertIn("Device mesh dimension size mismatch", str(cm.exception))

    def test_validate_device_mesh_success(self):
        """Test validate_device_mesh with valid mesh."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1) 

        mock_mesh = Mock()
        mock_mesh.size.return_value = 4
        mock_mesh.mesh_dim_names = ("dp_replicate", "dp_shard")
        mock_mesh.shape = (2, 2)
        
        config.validate_device_mesh(mock_mesh)

    def test_validate_device_mesh_with_tp(self):
        """Test validate_device_mesh with tensor parallelism."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=2)  
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 8
        mock_mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        mock_mesh.shape = (2, 2, 2)
        
        config.validate_device_mesh(mock_mesh)

    def test_validate_device_mesh_no_parallelism(self):
        """Test validate_device_mesh with no parallelism."""
        from unittest.mock import Mock
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1) 
        
        mock_mesh = Mock()
        mock_mesh.size.return_value = 1
        mock_mesh.mesh_dim_names = ()
        mock_mesh.shape = ()
        
        config.validate_device_mesh(mock_mesh)