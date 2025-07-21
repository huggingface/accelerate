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
from accelerate import Accelerator
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



class TestParallelismConfig():

    @pytest.fixture(autouse=True)
    def mock_init_device_mesh(self):
        def mock_init_mesh(device_type, mesh_shape, mesh_dim_names):
            mesh = Mock()
            mesh.size.return_value = 1
            for dim in mesh_shape:
                mesh.size.return_value *= dim
            mesh.shape = mesh_shape
            mesh.mesh_dim_names = mesh_dim_names
            return mesh
        
        with patch('torch.distributed.init_device_mesh', side_effect=mock_init_mesh):
            yield mock_init_mesh


    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, expected_shape, expected_dim_names",
        [
            (1, 1, 1, (), ()),
            (8, 1, 1, (8,), ("dp_replicate",)), # DDP
            (1, 8, 1, (8,), ("dp_shard",)), # FSDP
            (2, 4, 1, (2, 4), ("dp_replicate", "dp_shard")), # HSDP
            (1, 4, 2, (4, 2), ("dp_shard", "tp")), # FSDP + TP
            (2, 2, 2, (2, 2, 2), ("dp_replicate", "dp_shard", "tp")), # HSDP + TP
            (1, 1, 8, (8,), ("tp",)), # TP only
        ]
    )
    def test_get_mesh(
        self, 
        dp_replicate_size, 
        dp_shard_size, 
        tp_size, 
        expected_shape, 
        expected_dim_names,
    ):
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, 
            dp_shard_size=dp_shard_size, 
            tp_size=tp_size
        )
        mesh_dim_names, mesh_shape = config.get_mesh()
        assert mesh_shape ==  expected_shape
        assert mesh_dim_names ==  expected_dim_names


    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, device_mesh_dims, device_mesh_dim_names",
        [
            (8, 1, 1, (8,), ("dp_replicate",)),
            (1, 8, 1, (8,), ("dp_shard",)),
            (2, 4, 1, (2, 4), ("dp_replicate", "dp_shard")),
            (1, 4, 2, (4, 2), ("dp_shard", "tp")),
            (2, 2, 2, (2, 2, 2), ("dp_replicate", "dp_shard", "tp")),
            (1, 1, 8, (8,), ("tp",)),
        ]
    )
    def test_validate_device_mesh_valid_configurations(
        self, 
        dp_replicate_size, 
        dp_shard_size, 
        tp_size, 
        device_mesh_dims, 
        device_mesh_dim_names,
    ):
        """Test validate_device_mesh with valid configurations."""        
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, 
            dp_shard_size=dp_shard_size, 
            tp_size=tp_size
        )  
        device_mesh = dist.init_device_mesh("cpu", mesh_shape=device_mesh_dims, mesh_dim_names=device_mesh_dim_names)
        config.validate_device_mesh(device_mesh)


    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, device_mesh_dims, device_mesh_dim_names",
        [
            (8, 1, 1, (4,), ("dp_replicate",)), # invalid size
            (1, 4, 1, (4,), ("tp",)), # valid size, invalid mesh name
            (2, 4, 1, (2, 3), ("dp_replicate", "dp_shard")), # invalid total world size
            (4, 4, 2, (4, 4), ("dp_shard", "dp_replicate")), # valid size, invalid ordering
        ]
    )
    def test_validate_device_mesh_invalid_configurations(
        self, 
        dp_replicate_size, 
        dp_shard_size, 
        tp_size, 
        device_mesh_dims, 
        device_mesh_dim_names,
    ):
        """Test validate_device_mesh with invalid configurations."""        
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, 
            dp_shard_size=dp_shard_size, 
            tp_size=tp_size
        )  
        device_mesh = dist.init_device_mesh("cpu", mesh_shape=device_mesh_dims, mesh_dim_names=device_mesh_dim_names)
        with pytest.raises(ValueError):
            config.validate_device_mesh(device_mesh)

