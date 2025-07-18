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
import pytest
from unittest.mock import Mock, patch
import torch
import torch.distributed as dist

from accelerate.test_utils import (
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_multi_device,
    run_first,
    torch_device,
)

class TestParallelismConfig(AccelerateTestCase):

    @pytest.fixture()
    def device_mesh(self, mesh_dims, mesh_dim_names):
        torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
        torch_accelerator_module.set_device(device := torch.device(dist.get_rank()))
        mesh = dist.init_device_mesh(device.type, mesh_dims, mesh_dim_names)
        return mesh
        

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, expected_shape, expected_dim_names",
        [
            (1, 1, 1, (), ()),
            (8, 1, 1, (8,), ("dp_replicate",)), # DDP
            (1, 8, 1, (8,), ("dp_shard",)), # FSDP
            (2, 4, 1, (2, 4), ("dp_replicate", "dp_shard")), # HSDP
            (1, 4, 2, (4, 2), ("dp_shard", "tp")), # FSDP + TP
            (4, 1, 2, (4, 2), ("dp_replicate", "tp")), # DDP + TP
            (2, 2, 2, (2, 2, 2), ("dp_replicate", "dp_shard", "tp")), # HSDP + TP
            (1, 1, 8, (8,), ("tp",)), # TP only
        ]
    )
    def test_get_mesh(self, dp_replicate_size, dp_shard_size, tp_size, expected_shape, expected_dim_names):
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, 
            dp_shard_size=dp_shard_size, 
            tp_size=tp_size
        )
        mesh_shape, mesh_dim_names = config.get_mesh()
        self.assertEqual(mesh_shape, expected_shape)
        self.assertEqual(mesh_dim_names, expected_dim_names)

    def test_validate_accelerator_total_size_mismatch(self):
        """Test validate_accelerator with total_size mismatch."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=2) 
        
        accelerator = Accelerator() 

        with self.assertRaises(ValueError) as cm:
            config.validate_accelerator(accelerator)
        self.assertIn("requires 8 processes, but the current Accelerator is set to use 1 processes", str(cm.exception))

    def test_validate_accelerator_success(self):
        """Test validate_accelerator with matching config."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1) 

        
        accelerator = Accelerator()
        config.validate_accelerator(accelerator)

    
    def test_validate_device_mesh_size_mismatch(self):
        """Test validate_device_mesh with size mismatch."""        
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

    def test_accelerator_device_mesh_integration_no_parallelism(self):
        """Test accelerator device mesh integration with no parallelism."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1)
        accelerator = Accelerator(parallelism_config=config)
        
        # Should not create device mesh when no parallelism is enabled
        self.assertIsNone(accelerator.device_mesh)

    def test_accelerator_validation_total_size_mismatch(self):
        """Test accelerator validation fails when total_size doesn't match num_processes."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=2, dp_shard_size=2, tp_size=1)  # total_size = 4
        config._is_fully_initialized = True
        
        accelerator = Accelerator()  # This creates accelerator with num_processes = 1
        
        with self.assertRaises(ValueError) as cm:
            config.validate_accelerator(accelerator)
        self.assertIn("total_size (4) does not match num_processes (1)", str(cm.exception))

    def test_accelerator_validation_success_single_process(self):
        """Test accelerator validation succeeds with matching single process config."""
        from accelerate import Accelerator
        
        config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=1, tp_size=1)  # total_size = 1
        config._is_fully_initialized = True
        
        accelerator = Accelerator()
        # This should not raise any exception
        config.validate_accelerator(accelerator)

    def test_build_device_mesh_error_handling(self):
        """Test that device mesh creation failures are properly handled."""
        from accelerate import Accelerator
        from unittest.mock import patch
        
        # Mock torch.distributed.init_device_mesh to raise an exception
        with patch('torch.distributed.init_device_mesh') as mock_init_mesh:
            mock_init_mesh.side_effect = RuntimeError("Device mesh creation failed")
            
            config = ParallelismConfig(dp_replicate_size=1, dp_shard_size=2, tp_size=1)
            
            with self.assertRaises(RuntimeError) as cm:
                accelerator = Accelerator(parallelism_config=config)
            
            self.assertIn("Failed to create device mesh with shape", str(cm.exception))
            self.assertIn("Device mesh creation failed", str(cm.exception))