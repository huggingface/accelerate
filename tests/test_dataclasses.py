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

from unittest.mock import Mock, patch

import pytest

from accelerate.parallelism_config import ParallelismConfig


class TestParallelismConfig:
    @pytest.fixture(autouse=True)
    def mock_init_device_mesh(self):
        def mock_init_mesh(device_type, mesh_shape, mesh_dim_names):
            mesh = Mock()
            mesh.size.return_value = 1
            for dim in mesh_shape:
                mesh.size.return_value *= dim
            mesh.shape = mesh_shape
            mesh.mesh_dim_names = mesh_dim_names

            # mock device_mesh._flatten
            mesh.flattened_dims = []

            def mock_getitem(key):
                submesh = Mock()

                def mock_flatten(name):
                    mesh.flattened_dims.append((key, name))

                submesh._flatten = Mock(side_effect=mock_flatten)
                return submesh

            mesh.__getitem__ = Mock(side_effect=mock_getitem)

            return mesh

        with patch("torch.distributed.init_device_mesh", side_effect=mock_init_mesh):
            yield mock_init_mesh

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, cp_size, expected_shape, expected_dim_names",
        [
            (8, 1, 1, 1, (8,), ("dp_replicate",)),  # DDP
            (1, 8, 1, 1, (8,), ("dp_shard",)),  # FSDP
            (2, 4, 1, 1, (2, 4), ("dp_replicate", "dp_shard")),  # HSDP
            (1, 4, 2, 1, (4, 2), ("dp_shard", "tp")),  # FSDP + TP
            (2, 2, 2, 1, (2, 2, 2), ("dp_replicate", "dp_shard", "tp")),  # HSDP + TP
            (1, 1, 8, 1, (8,), ("tp",)),  # TP only
            (1, 1, 1, 4, (4,), ("cp",)),  # CP only
            (1, 4, 1, 2, (4, 2), ("dp_shard", "cp")),  # FSDP + CP
            (1, 2, 2, 2, (2, 2, 2), ("dp_shard", "cp", "tp")),  # FSDP + CP + TP
            (2, 2, 2, 2, (2, 2, 2, 2), ("dp_replicate", "dp_shard", "cp", "tp")),  # HSDP + CP + TP
        ],
    )
    def test_get_mesh(
        self,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        cp_size,
        expected_shape,
        expected_dim_names,
    ):
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, dp_shard_size=dp_shard_size, tp_size=tp_size, cp_size=cp_size
        )
        mesh_dim_names, mesh_shape = config._get_mesh()
        assert mesh_shape == expected_shape
        assert mesh_dim_names == expected_dim_names

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, cp_size, expected_shape, expected_dim_names",
        [
            (8, 1, 1, 1, (8,), ("dp_replicate",)),
            (1, 8, 1, 1, (8,), ("dp_shard",)),
            (2, 4, 1, 1, (2, 4), ("dp_replicate", "dp_shard")),
            (1, 4, 2, 1, (4, 2), ("dp_shard", "tp")),
            (2, 2, 2, 1, (2, 2, 2), ("dp_replicate", "dp_shard", "tp")),
            (1, 1, 8, 1, (8,), ("tp",)),
            (1, 1, 1, 4, (4,), ("cp",)),
            (1, 4, 1, 2, (4, 2), ("dp_shard", "cp")),
            (1, 2, 2, 2, (2, 2, 2), ("dp_shard", "cp", "tp")),
            (2, 2, 2, 2, (2, 2, 2, 2), ("dp_replicate", "dp_shard", "cp", "tp")),
        ],
    )
    def test_build_device_mesh(
        self,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        cp_size,
        expected_shape,
        expected_dim_names,
    ):
        """Test build_device_mesh creates correct mesh and applies flattening."""
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size, dp_shard_size=dp_shard_size, tp_size=tp_size, cp_size=cp_size
        )
        device_mesh = config.build_device_mesh("cpu")

        # Check mesh shape and dimension names match expected
        assert device_mesh.shape == expected_shape
        assert device_mesh.mesh_dim_names == expected_dim_names

        # Check that correct flattening operations were called
        expected_flattened = []
        if config.dp_dim_names:
            expected_flattened.append((config.dp_dim_names, "dp"))
        if config.dp_shard_cp_dim_names:
            expected_flattened.append((config.dp_shard_cp_dim_names, "dp_shard_cp"))
        if config.dp_cp_dim_names:
            expected_flattened.append((config.dp_cp_dim_names, "dp_cp"))

        assert device_mesh.flattened_dims == expected_flattened
