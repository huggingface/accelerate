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
from accelerate.utils import patch_environment
from accelerate.utils.constants import (
    BETA_CP_AVAILABLE_PYTORCH_VERSION,
    BETA_SP_AVAILABLE_DEEPSPEED_VERSION,
    BETA_TP_AVAILABLE_PYTORCH_VERSION,
    BETA_TP_AVAILABLE_TRANSFORMERS_VERSION,
)
from accelerate.utils.imports import is_deepspeed_available, is_transformers_available
from accelerate.utils.versions import compare_versions, is_torch_version


def _should_skip_cp_test(cp_size):
    """Check if CP test should be skipped based on cp_size and torch version."""
    return cp_size > 1 and not is_torch_version(">=", BETA_CP_AVAILABLE_PYTORCH_VERSION)


def _should_skip_sp_test(sp_size):
    """Check if SP test should be skipped based on sp_size and deepspeed version."""
    if sp_size <= 1:
        return False
    if not is_deepspeed_available():
        return True
    return not compare_versions("deepspeed", ">=", BETA_SP_AVAILABLE_DEEPSPEED_VERSION)


def _should_skip_tp_test(tp_size):
    """Check if TP test should be skipped based on tp_size, torch version, and transformers availability."""
    if tp_size <= 1:
        return False

    if not is_torch_version(">=", BETA_TP_AVAILABLE_PYTORCH_VERSION):
        return True

    if not is_transformers_available():
        return True

    if not compare_versions("transformers", ">=", BETA_TP_AVAILABLE_TRANSFORMERS_VERSION):
        return True

    return False


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

        with patch("torch.distributed.device_mesh.init_device_mesh", side_effect=mock_init_mesh):
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
        # Skip tests based on version requirements
        if _should_skip_cp_test(cp_size):
            pytest.skip(f"tests with `cp_size>1` require torch >= {BETA_CP_AVAILABLE_PYTORCH_VERSION}")
        if _should_skip_tp_test(tp_size):
            pytest.skip(
                f"tests with `tp_size>1` require torch >= {BETA_TP_AVAILABLE_PYTORCH_VERSION}, transformers available and >= {BETA_TP_AVAILABLE_TRANSFORMERS_VERSION}"
            )

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
        # Skip tests based on version requirements
        if _should_skip_cp_test(cp_size):
            pytest.skip(f"tests with `cp_size>1` require torch >= {BETA_CP_AVAILABLE_PYTORCH_VERSION}")
        if _should_skip_tp_test(tp_size):
            pytest.skip(
                f"tests with `tp_size>1` require torch >= {BETA_TP_AVAILABLE_PYTORCH_VERSION}, transformers available and >= {BETA_TP_AVAILABLE_TRANSFORMERS_VERSION}"
            )

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

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, cp_size",
        [
            (8, 1, 1, 1),
            (1, 8, 1, 1),
            (2, 4, 1, 1),
            (1, 4, 2, 1),
            (2, 2, 2, 1),
            (1, 1, 8, 1),
            (1, 1, 1, 4),
            (1, 4, 1, 2),
            (1, 2, 2, 2),
            (2, 2, 2, 2),
        ],
    )
    def test_from_env(
        self,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        cp_size,
    ):
        if _should_skip_cp_test(cp_size):
            pytest.skip(f"tests with `cp_size>1` require torch >= {BETA_CP_AVAILABLE_PYTORCH_VERSION}")
        if _should_skip_tp_test(tp_size):
            pytest.skip(
                f"tests with `tp_size>1` require torch >= {BETA_TP_AVAILABLE_PYTORCH_VERSION}, transformers available and >= {BETA_TP_AVAILABLE_TRANSFORMERS_VERSION}"
            )

        new_env = {
            "PARALLELISM_CONFIG_DP_REPLICATE_SIZE": dp_replicate_size,
            "PARALLELISM_CONFIG_DP_SHARD_SIZE": dp_shard_size,
            "PARALLELISM_CONFIG_TP_SIZE": tp_size,
            "PARALLELISM_CONFIG_CP_SIZE": cp_size,
            "PARALLELISM_CONFIG_EP_SIZE": 1,
        }

        with patch_environment(**new_env):
            config = ParallelismConfig()
            for key, value in new_env.items():
                assert getattr(config, key.split("PARALLELISM_CONFIG_")[-1].lower()) == value

    def test_cp_torch_handler(self):
        """Test CP Torch/FSDP2 handler with various configurations."""

        # Any cp_size > 1 requires torch >= BETA_CP_AVAILABLE_PYTORCH_VERSION, we use placeholder for this check as this test doesn't depend on a specific size
        if _should_skip_cp_test(2):
            pytest.skip(f"tests with `cp_size>1` require torch >= {BETA_CP_AVAILABLE_PYTORCH_VERSION}")

        from accelerate.utils import TorchContextParallelConfig

        for setting in ("allgather", "alltoall"):
            cp_handler = TorchContextParallelConfig(cp_comm_strategy=setting)
            pc = ParallelismConfig(cp_size=2, cp_handler=cp_handler)

            assert pc.cp_handler is not None, "CP handler should be set"
            assert pc.cp_handler.cp_comm_strategy == setting, (
                f"CP handler strategy should be {setting} but got {pc.cp_handler.cp_comm_strategy}"
            )

        for setting in ("allgather", "alltoall"):
            with patch_environment(PARALLELISM_CONFIG_CP_COMM_STRATEGY=setting):
                pc = ParallelismConfig(cp_size=2)
                assert pc.cp_handler is not None, "CP handler should be set from environment"
                assert pc.cp_handler.cp_comm_strategy == setting, (
                    f"CP handler strategy should be {setting} but got {pc.cp_handler.cp_comm_strategy}"
                )

        for setting in ("invalid", "unsupported"):
            with pytest.raises(ValueError, match=f"Invalid cp_comm_strategy: {setting}"):
                TorchContextParallelConfig(cp_comm_strategy=setting)

            with patch_environment(PARALLELISM_CONFIG_CP_COMM_STRATEGY=setting):
                with pytest.raises(ValueError, match=f"Invalid cp_comm_strategy: {setting}"):
                    pc = ParallelismConfig(cp_size=2)

    def test_sp_deepspeed_handler(self):
        """Test SP DeepSpeed/ALST/UlyssesSP handler with various configurations."""

        # Any sp_size > 1 requires torch >= BETA_SP_AVAILABLE_PYTORCH_VERSION, we use placeholder for this check as this test doesn't depend on a specific size
        if _should_skip_sp_test(2):
            pytest.skip(f"tests with `sp_size>1` require deepspeed >= {BETA_SP_AVAILABLE_DEEPSPEED_VERSION}")

        from accelerate.utils import DeepSpeedSequenceParallelConfig

        sp_handler = DeepSpeedSequenceParallelConfig()
        pc = ParallelismConfig(sp_backend="deepspeed", sp_size=2, sp_handler=sp_handler)
        assert pc.sp_handler is not None, "SP handler should be set"
        assert pc.sp_handler.sp_seq_length_is_variable is True, "by default we set to expect a variable seqlen"

        with pytest.raises(ValueError, match="Invalid sp_attn_implementation"):
            DeepSpeedSequenceParallelConfig(sp_attn_implementation="foobar")

    def test_tp_handler(self):
        assert True, "Tensor parallelism handler doesn't hold any logic yet"

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, ep_size, expected_shape, expected_dim_names",
        [
            (1, 1, 1, 4, (4,), ("ep",)),  # EP only
            (1, 2, 1, 4, (2, 4), ("dp_shard", "ep")),  # FSDP + EP
            (1, 2, 2, 2, (2, 2, 2), ("dp_shard", "ep", "tp")),  # FSDP + EP + TP
            (2, 2, 1, 2, (2, 2, 2), ("dp_replicate", "dp_shard", "ep")),  # HSDP + EP
        ],
    )
    def test_get_mesh_with_ep(
        self,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        ep_size,
        expected_shape,
        expected_dim_names,
    ):
        # Dummy handler avoids TorchTensorParallelConfig version checks; we only assert mesh layout.
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            tp_size=tp_size,
            ep_size=ep_size,
            tp_handler=Mock() if tp_size > 1 else None,
        )
        mesh_dim_names, mesh_shape = config._get_mesh()
        assert mesh_shape == expected_shape
        assert mesh_dim_names == expected_dim_names

    @pytest.mark.parametrize(
        "dp_replicate_size, dp_shard_size, tp_size, ep_size, expected_shape, expected_dim_names",
        [
            (1, 1, 1, 4, (4,), ("ep",)),
            (1, 2, 1, 4, (2, 4), ("dp_shard", "ep")),
            (1, 2, 2, 2, (2, 2, 2), ("dp_shard", "ep", "tp")),
            (2, 2, 1, 2, (2, 2, 2), ("dp_replicate", "dp_shard", "ep")),
        ],
    )
    def test_build_device_mesh_with_ep(
        self,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        ep_size,
        expected_shape,
        expected_dim_names,
    ):
        """EP must appear on the mesh but must not be flattened into FSDP dims."""
        config = ParallelismConfig(
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            tp_size=tp_size,
            ep_size=ep_size,
            tp_handler=Mock() if tp_size > 1 else None,
        )
        device_mesh = config.build_device_mesh("cpu")

        assert device_mesh.shape == expected_shape
        assert device_mesh.mesh_dim_names == expected_dim_names

        expected_flattened = []
        if config.dp_dim_names:
            expected_flattened.append((config.dp_dim_names, "dp"))
        if config.dp_shard_cp_dim_names:
            expected_flattened.append((config.dp_shard_cp_dim_names, "dp_shard_cp"))
        if config.dp_cp_dim_names:
            expected_flattened.append((config.dp_cp_dim_names, "dp_cp"))

        assert device_mesh.flattened_dims == expected_flattened
        for flattened_keys, flattened_name in device_mesh.flattened_dims:
            assert "ep" not in flattened_keys
            assert flattened_name != "ep"
        assert "ep" not in config.fsdp_dim_names
        assert "ep" not in config.dp_shard_cp_dim_names
        assert "ep" not in config.dp_cp_dim_names
        assert "ep" not in config.dp_dim_names

    def test_from_env_ep_size(self):
        with patch_environment(PARALLELISM_CONFIG_EP_SIZE=4):
            config = ParallelismConfig()
            assert config.ep_size == 4
            assert config.ep_enabled

    def test_ep_size_zero_raises(self):
        with pytest.raises(ValueError, match="ep_size must be at least 1"):
            ParallelismConfig(ep_size=0)

    def test_ep_dim_names_and_sizes(self):
        config = ParallelismConfig(dp_shard_size=2, ep_size=4)

        assert config.ep_enabled
        assert "ep" in config.non_dp_dim_names
        assert "ep" not in config.dp_dim_names
        assert "ep" not in config.fsdp_dim_names
        assert "ep" not in config.dp_shard_cp_dim_names
        assert "ep" not in config.dp_cp_dim_names
        assert config.total_size == 8
        assert config.non_data_parallel_size == 4
        assert config.data_parallel_size == 2

    def test_default_ep_size_does_not_change_existing_mesh(self):
        if _should_skip_cp_test(2):
            pytest.skip(f"tests with `cp_size>1` require torch >= {BETA_CP_AVAILABLE_PYTORCH_VERSION}")
        config = ParallelismConfig(dp_shard_size=2, cp_size=2)
        mesh_dim_names, mesh_shape = config._get_mesh()
        assert config.ep_size == 1
        assert not config.ep_enabled
        assert "ep" not in mesh_dim_names
        assert mesh_shape == (2, 2)
        assert mesh_dim_names == ("dp_shard", "cp")
        assert config.total_size == 4
        assert config.non_data_parallel_size == 2

    def test_ep_with_dp_replicate_is_allowed(self):
        """Replicating an expert-parallel model is a valid layout; the TP/CP DDP restriction does not apply to EP."""
        config = ParallelismConfig(dp_replicate_size=2, ep_size=2)
        mesh_dim_names, mesh_shape = config._get_mesh()
        assert mesh_shape == (2, 2)
        assert mesh_dim_names == ("dp_replicate", "ep")
        assert config.total_size == 4

    def test_ep_same_batch_rank_mapping(self):
        """EP ranks receive the same batch, like TP/CP: process_index // (tp * cp * ep)."""
        config = ParallelismConfig(dp_shard_size=2, ep_size=4)
        mesh_dim_names, mesh_shape = config._get_mesh()
        assert mesh_dim_names == ("dp_shard", "ep")
        assert mesh_shape == (2, 4)

        sizes = dict(zip(mesh_dim_names, mesh_shape))
        submesh_tp_size = sizes.get("tp", 1)
        submesh_cp_size = sizes.get("cp", 1)
        submesh_ep_size = sizes.get("ep", 1)
        submesh_fsdp_size = sizes.get("dp_shard", 1)
        submesh_dp_size = sizes.get("dp_replicate", 1)

        remapped_indices = [
            process_index // (submesh_tp_size * submesh_cp_size * submesh_ep_size) for process_index in range(8)
        ]
        assert remapped_indices == [0, 0, 0, 0, 1, 1, 1, 1]
        assert submesh_fsdp_size * submesh_dp_size == 2
