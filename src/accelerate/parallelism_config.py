#
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

from accelerate.utils.dataclasses import (
    DeepSpeedSequenceParallelConfig,
    DistributedType,
    TorchContextParallelConfig,
    AccelerateSequenceParallelConfig,
    TorchTensorParallelConfig,
)
from accelerate.utils.versions import is_torch_version


if TYPE_CHECKING:
    from accelerate import Accelerator


@dataclass
class ParallelismConfig:
    """
    A dataclass to configure parallelisms applied to the model. Inspired by torchtitan's `ParallelDims`
    https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

    Args:
        dp_replicate_size (`int`, defaults to `1`):
            The size of the data parallel group. If `dp_replicate_size` is set to 1, the data parallel replication
            group will not be used.
        dp_shard_size (`int`, defaults to `1`):
            The size of the model shard group. If `dp_replicate_size > 1` and `tp_size > 1`, `dp_shard_size` must also
            be greater than 1, as composing DDP + TP is currently not supported.
        tp_size (`int`, defaults to `1`):
            The size of the tensor parallel group. If `tp_size` is set to `1`, the tensor parallel group will not be
            used.
        tp_handler (`~utils.TorchTensorParallelConfig`, defaults to `None`):
            The handler for the tensor parallel group.
        cp_size (`int`, defaults to `1`):
            The size of the context parallel group. Currently not supported, but reserved for future use and enabled
            for downstream libraries.
        cp_backend (`str`, defaults to `torch`):
            Which CP backend to use: `torch` (FSDP2)
        cp_handler (`~utils.TorchContextParallelConfig`, *optional*,  defaults to `None`): The handler for the context parallel group.
        sp_size (`int`, defaults to `1`):
            The size of the sequence parallel group. If `1`, SP is not used.
        sp_backend (`str`, *optional*):
            Which SP backend to use: `"accelerate"` (our native Ulysses) or `"deepspeed"`
            (ALST/Ulysses, which needs the DeepSpeed engine). If left unset it auto-resolves: `"deepspeed"`
            under the DeepSpeed engine, else `"accelerate"`.
        sp_handler (`~utils.AccelerateSequenceParallelConfig` or `~utils.DeepSpeedSequenceParallelConfig`, *optional*, defaults to `None`):
            Config for the sequence parallel group;

    You may obtain different distributed data parallel paradigms by configuring `dp_replicate_size` and `dp_shard_size`
    together:
        - `dp_replicate_size == 1` and `dp_shard_size > 1`, we obtain Fully Sharded Data Parallel (FSDP).
        - `dp_replicate_size > 1` and `dp_shard_size > 1`, we obtain Hybrid Sharded Data Parallel (HSDP).
        - `dp_replicate_size > 1` and `dp_shard_size == 1` is an invalid configuration, to use pure DP, use
          `DistributedDataParallelKwargs` instead.

    """

    dp_replicate_size: Optional[int] = None
    dp_shard_size: Optional[int] = None
    tp_size: Optional[int] = None
    cp_size: Optional[int] = None
    cp_backend: Literal["torch"] = None
    sp_size: Optional[int] = None
    sp_backend: Literal["deepspeed", "accelerate"] = None

    # we use Union because we might support other x parallel plugins (i.e. deepspeed, etc)
    tp_handler: Union[None, TorchTensorParallelConfig] = None
    cp_handler: Union[None, TorchContextParallelConfig] = None
    sp_handler: Union[None, DeepSpeedSequenceParallelConfig, AccelerateSequenceParallelConfig] = None

    device_mesh = None
    # Flattened submeshes (dp / dp_shard_cp / dp_cp) returned by `_flatten`, kept so consumers don't
    # re-slice them off the root mesh (deprecated in torch >= 2.11). Populated by `build_device_mesh`.
    _flattened_meshes = None

    def __repr__(self):
        return (
            "ParallelismConfig(\n "
            f"\tdp_replicate_size={self.dp_replicate_size},\n"
            f"\tdp_shard_size={self.dp_shard_size},\n"
            f"\ttp_size={self.tp_size},\n"
            f"\tcp_size={self.cp_size},\n"
            f"\tcp_backend={self.cp_backend},\n"
            f"\tsp_size={self.sp_size},\n"
            f"\tsp_backend={self.sp_backend},\n"
            f"\ttotal_size={self.total_size}\n"
            f"\ttp_handler={self.tp_handler},\n"
            f"\tcp_handler={self.cp_handler},\n"
            f"\tsp_handler={self.sp_handler})\n"
        )

    def to_json(self):
        import copy

        _non_serializable_fields = ["device_mesh"]

        return copy.deepcopy(
            {
                k: copy.deepcopy(v.__dict__) if hasattr(v, "__dict__") else v
                for k, v in self.__dict__.items()
                if k not in _non_serializable_fields
            }
        )

    @property
    def dp_dim_names(self):
        """Names of enabled dimensions across which data parallelism is applied."""
        dims = []
        if self.dp_replicate_enabled:
            dims += ["dp_replicate"]
        if self.dp_shard_enabled:
            dims += ["dp_shard"]
        return dims

    @property
    def non_dp_dim_names(self):
        """Names of enabled dimensions which will receive the same batch (non-data parallel dimensions)."""
        dims = []
        if self.tp_enabled:
            dims += ["tp"]
        if self.cp_enabled:
            dims += ["cp"]
        if self.sp_enabled:
            dims += ["sp"]
        return dims

    @property
    def _sequence_sharding_enabled(self):
        """Whether the native (accelerate) sequence-sharding path is active and `prepare` should wrap
        dataloaders in a `SequenceShardingDataLoader`. Currently Ulysses SP on the accelerate backend;
        ring CP will OR its clause in here once added."""
        return self.sp_enabled and self.sp_backend == "accelerate"

    @property
    def dp_shard_cp_dim_names(self):
        """The FSDP shard axis — flattened into one mesh dim (`dp_shard_cp`, accelerate's name) that
        `fully_shard` shards params / reduce-scatters grads over. Despite the name it spans `dp_shard`
        **plus the sequence-parallel dims `cp` and `sp`**: both shard the same sample's sequence, so
        FSDP must shard over them for its reduce-scatter to average their grads."""
        dims = []
        if self.dp_shard_enabled:
            dims += ["dp_shard"]
        if self.cp_enabled:
            dims += ["cp"]
        if self.sp_enabled:
            dims += ["sp"]
        return dims

    @property
    def dp_cp_dim_names(self):
        """Dims over which loss/grads are averaged, flattened to the `dp_cp` mesh dim. Despite the
        name it spans the data-parallel dims (`dp_replicate`, `dp_shard`) **plus the sequence-parallel
        dims `cp` and `sp`**."""
        dims = []
        if self.dp_replicate_enabled:
            dims += ["dp_replicate"]
        if self.dp_shard_enabled:
            dims += ["dp_shard"]
        if self.cp_enabled:
            dims += ["cp"]
        if self.sp_enabled:
            dims += ["sp"]
        return dims

    @property
    def fsdp_dim_names(self):
        """Names of enabled dimensions across which FSDP is applied, including data parallel replication."""
        dims = []
        if self.dp_replicate_enabled:
            dims += ["dp_replicate"]
        dims += ["dp_shard_cp"]
        return dims

    @property
    def total_size(self):
        """The total size of the parallelism configuration, which is the product of all sizes."""
        return self.dp_replicate_size * self.dp_shard_size * self.tp_size * self.cp_size * self.sp_size

    @property
    def non_data_parallel_size(self):
        """Product of the non-data-parallel sizes (tensor x context/ring x sequence/Ulysses)."""
        return self.tp_size * self.cp_size * self.sp_size

    @property
    def data_parallel_size(self):
        """Product of the data-parallel sizes (replication x shard)."""
        return self.dp_replicate_size * self.dp_shard_size

    @property
    def dp_replicate_enabled(self):
        """True if data parallel replication is enabled, i.e. `dp_replicate_size > 1`."""
        return self.dp_replicate_size > 1

    @property
    def dp_shard_enabled(self):
        """True if data parallel sharding is enabled, i.e. `dp_shard_size > 1`."""
        return self.dp_shard_size > 1

    @property
    def tp_enabled(self):
        """True if tensor parallelism is enabled, i.e. `tp_size > 1`."""
        return self.tp_size > 1

    @property
    def cp_enabled(self):
        """True if context parallelism is enabled, i.e. `cp_size > 1`."""
        return self.cp_size > 1

    @property
    def sp_enabled(self):
        """True if sequence parallelism is enabled, i.e. `sp_size > 1`."""
        return self.sp_size > 1

    @property
    def active_mesh_dims(self):
        """Names of all active mesh dimensions."""
        return self.dp_dim_names + self.non_dp_dim_names

    def build_device_mesh(self, device_type: str):
        """Builds a device mesh for the given device type based on the parallelism configuration.
        This method will also create required joint meshes (e.g. `dp_shard_cp`, `dp_cp`, `dp`).

        Args:
            device_type (`str`): The type of device for which to build the mesh, e
        """
        # Skip mesh creation for DeepSpeed SP - DeepSpeed handles its own SP groups
        # Only skip when SP is actually enabled (sp_size > 1), otherwise user might still want TP/CP/FSDP
        if self.sp_backend == "deepspeed" and self.sp_size > 1:
            return None

        if is_torch_version(">=", "2.2.0"):
            from torch.distributed.device_mesh import init_device_mesh
        else:
            raise RuntimeError("Building a device_mesh requires to have torch>=2.2.0")

        mesh = self._get_mesh()
        if len(mesh) == 0:
            return None
        mesh_dim_names, mesh_shape = mesh
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        # Keep the flattened submeshes that `_flatten` returns. Re-slicing a flattened dim off the root
        # mesh (e.g. `device_mesh["dp_shard_cp"]`) is deprecated in torch >= 2.11, so consumers read them
        # back through `get_submesh` instead.
        self._flattened_meshes = {}
        if self.dp_dim_names:
            self._flattened_meshes["dp"] = device_mesh[self.dp_dim_names]._flatten("dp")
        if self.dp_shard_cp_dim_names:
            self._flattened_meshes["dp_shard_cp"] = device_mesh[self.dp_shard_cp_dim_names]._flatten("dp_shard_cp")
        if self.dp_cp_dim_names:
            self._flattened_meshes["dp_cp"] = device_mesh[self.dp_cp_dim_names]._flatten("dp_cp")

        return device_mesh

    def get_device_mesh(self, device_type: Optional[str] = None):
        if self.device_mesh is None:
            if device_type is not None:
                self.device_mesh = self.build_device_mesh(device_type)
            else:
                raise ValueError("You need to pass a device_type e.g cuda to build the device mesh")
        else:
            if device_type is not None:
                if self.device_mesh.device_type != device_type:
                    raise ValueError(
                        f"The device_mesh is already created with device type {self.device_mesh.device_type}. However, you are trying to get a device mesh with device_type {device_type}. Please check if you correctly initialized your device_mesh"
                    )
        return self.device_mesh

    def get_submesh(self, dim_names):
        """Return the submesh for `dim_names`. A single flattened dim (`dp` / `dp_shard_cp` / `dp_cp`) is
        returned from the handle stored at build time, since re-slicing a flattened dim off the root mesh
        is deprecated in torch >= 2.11. Anything else falls through to normal mesh indexing."""
        dim_names = tuple(dim_names)
        flattened = self._flattened_meshes or {}
        if len(dim_names) == 1 and dim_names[0] in flattened:
            return flattened[dim_names[0]]
        return self.device_mesh[dim_names]

    def _get_mesh(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Generate mesh shape and dimension names for torch.distributed.init_device_mesh()."""

        # Build mesh dimensions dictionary
        mesh_dims = {parallelism: self._sizes[parallelism] for parallelism in self.active_mesh_dims}

        # Apply canonical ordering
        mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
        sorted_items = sorted(
            mesh_dims.items(),
            key=lambda x: (mesh_order.index(x[0])),
        )
        return tuple(zip(*sorted_items))

    def __post_init__(self):
        # Track whether the user explicitly chose a backend (vs. leaving it to auto-resolve from
        # the training engine in `_resolve_backends`). Env-set counts as explicit. Must be read
        # BEFORE the defaulting below overwrites None.
        self._sp_backend_explicit = self.sp_backend is not None or "PARALLELISM_CONFIG_SP_BACKEND" in os.environ

        # Basic size validation
        if self.dp_replicate_size is None:
            self.dp_replicate_size = int(os.environ.get("PARALLELISM_CONFIG_DP_REPLICATE_SIZE", "1"))
        if self.dp_shard_size is None:
            self.dp_shard_size = int(os.environ.get("PARALLELISM_CONFIG_DP_SHARD_SIZE", "1"))
        if self.tp_size is None:
            self.tp_size = int(os.environ.get("PARALLELISM_CONFIG_TP_SIZE", "1"))
        if self.cp_size is None:
            self.cp_size = int(os.environ.get("PARALLELISM_CONFIG_CP_SIZE", "1"))
        if self.sp_size is None:
            self.sp_size = int(os.environ.get("PARALLELISM_CONFIG_SP_SIZE", "1"))
        if self.cp_backend is None:
            self.cp_backend = os.environ.get("PARALLELISM_CONFIG_CP_BACKEND", "torch")
        if self.sp_backend is None:
            self.sp_backend = os.environ.get("PARALLELISM_CONFIG_SP_BACKEND", "deepspeed")

        if self.tp_size > 1:
            if self.tp_handler is None:
                self.tp_handler = TorchTensorParallelConfig()

        if self.cp_size > 1:
            if self.cp_handler is None:
                self.cp_handler = TorchContextParallelConfig()
            else:
                cp_backends_config_map = dict(
                    torch=TorchContextParallelConfig,
                )
                if not isinstance(self.cp_handler, cp_backends_config_map[self.cp_backend]):
                    raise ValueError(
                        f"ParallelismConfig's cp_backend={self.cp_backend} requires {cp_backends_config_map[self.cp_backend]}, but cp_handler was set to {type(self.cp_handler)}"
                    )

        if self.sp_size > 1:
            if self.sp_handler is None:
                self.sp_handler = (
                    AccelerateSequenceParallelConfig()
                    if self.sp_backend == "accelerate"
                    else DeepSpeedSequenceParallelConfig()
                )
            else:
                sp_backends_config_map = dict(
                    deepspeed=DeepSpeedSequenceParallelConfig,
                    accelerate=AccelerateSequenceParallelConfig,
                )
                if not isinstance(self.sp_handler, sp_backends_config_map[self.sp_backend]):
                    raise ValueError(
                        f"ParallelismConfig's sp_backend={self.sp_backend} requires "
                        f"{sp_backends_config_map[self.sp_backend]}, but sp_handler was set to {type(self.sp_handler)}"
                    )
        if self.dp_replicate_size < 1:
            raise ValueError(f"dp_replicate_size must be at least 1, but got {self.dp_replicate_size}")
        if self.dp_shard_size < 1:
            raise ValueError(f"dp_shard_size must be at least 1, but got {self.dp_shard_size}")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be at least 1, but got {self.tp_size}")
        if self.cp_size < 1:
            raise ValueError(f"cp_size must be at least 1, but got {self.cp_size}")
        valid_cp_backends = ["torch"]
        if self.cp_backend not in valid_cp_backends:
            raise ValueError(f"cp_backend must be one of {valid_cp_backends}, but got {self.cp_backend}")

        if self.sp_size < 1:
            raise ValueError(f"sp_size must be at least 1, but got {self.sp_size}")
        valid_sp_backends = ["deepspeed", "accelerate"]
        if self.sp_backend not in valid_sp_backends:
            raise ValueError(f"sp_backend must be one of {valid_sp_backends}, but got {self.sp_backend}")

        # CP and SP are mutually exclusive
        if self.cp_size > 1 and self.sp_size > 1:
            raise ValueError(
                "Context Parallelism (CP) and Sequence Parallelism (SP) are mutually exclusive. "
                f"Got cp_size={self.cp_size} and sp_size={self.sp_size}. "
                "Please set either cp_size=1 or sp_size=1."
            )

        if (self.tp_size > 1 or self.cp_size > 1) and self.dp_replicate_size > 1 and self.dp_shard_size == 1:
            raise ValueError(
                "Tensor/Context parallelism (tp/cp_size > 1) cannot be used with pure data parallelism (dp_replicate_size > 1 and dp_shard_size == 1). "
                "Please set dp_shard_size > 1 and dp_replicate_size == 1 to compose FSDP + TP/CP for 2D parallel, "
                "or set dp_replicate_size == 1 and dp_shard_size > 1 to compose HSDP + TP/CP for 3D parallel."
            )
        self._sizes = {
            "dp_replicate": self.dp_replicate_size,
            "dp_shard": self.dp_shard_size,
            "tp": self.tp_size,
            "cp": self.cp_size,
            "sp": self.sp_size,
        }

    def _set_size(self, parallelism: str, size: int):
        assert parallelism in self._sizes.keys(), f"Parallelism must be one of {self._sizes.keys()}"
        self._sizes[parallelism] = size
        setattr(self, f"{parallelism}_size", size)

    def _resolve_backends(self, accelerator: "Accelerator"):
        """Resolve the sp (Ulysses) backend against the training engine, once ``distributed_type``
        is known. ``"deepspeed"`` (ALST) only runs under the DeepSpeed engine; ``"accelerate"``
        (native Ulysses) runs under any engine (DeepSpeed ZeRO, FSDP2, or DDP). When the user
        didn't pin a backend it defaults to ``"deepspeed"`` under DeepSpeed and ``"accelerate"``
        otherwise; an explicit ``"deepspeed"`` without the DeepSpeed engine falls back to
        ``"accelerate"``. ``sp_handler`` is set to match the resolved backend."""
        if self.sp_size <= 1:
            return

        is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED

        if not self._sp_backend_explicit:
            # ALST is the purpose-built path under DeepSpeed; native Ulysses under FSDP2 / DDP.
            self.sp_backend = "deepspeed" if is_deepspeed else "accelerate"
        elif self.sp_backend == "deepspeed" and not is_deepspeed:
            # ALST needs the DeepSpeed engine; fall back to native Ulysses (runs under FSDP2 / DDP).
            self.sp_backend = "accelerate"

        handler_cls = (
            DeepSpeedSequenceParallelConfig
            if self.sp_backend == "deepspeed"
            else AccelerateSequenceParallelConfig
        )
        if not isinstance(self.sp_handler, handler_cls):
            self.sp_handler = handler_cls()

    def _validate_accelerator(self, accelerator: "Accelerator"):
        _warnings = set()
        # Resolve auto backends + enforce backend<->engine compatibility now that the engine is known.
        self._resolve_backends(accelerator)
        if not accelerator.multi_device and self.total_size == 1:
            # No distributed setup, valid parallelism config
            return

        # We need this to ensure DDP works
        if self.total_size == 1:
            self._set_size("dp_replicate", accelerator.num_processes)

        # For DeepSpeed SP, DeepSpeed handles global process groups internally.
        # Skip the total_size == num_processes validation since:
        # 1. DeepSpeed manages SP groups globally via initialize_sequence_parallel()
        # 2. num_processes is per-node in multi-node, but total_size is local parallelism config
        # 3. The actual global parallelism (SP × DP) is handled by DeepSpeed's process groups
        if self.sp_backend == "deepspeed" and self.sp_size > 1:
            pass
        elif self.total_size != accelerator.num_processes:
            raise ValueError(
                f"ParallelismConfig total_size ({self.total_size}) does not match "
                f"num_processes ({accelerator.num_processes}). Please adjust dp_replicate_size/ "
                f"dp_shard_size/tp_size/cp_size/sp_size."
            )

        if self.total_size > 1 and not (
            accelerator.is_fsdp2
            or accelerator.multi_device
            or accelerator.distributed_type == DistributedType.DEEPSPEED
        ):
            raise ValueError(
                f"ParallelismConfig is only compatible DistributedType.FSDP (version 2) or DistributedType.Multi{{Device}} or DistributedType.DEEPSPEED, but got {accelerator.distributed_type}."
            )

        for parallelism, size in self._sizes.items():
            if size == 1 and getattr(self, f"{parallelism}_handler", None) is not None:
                _warnings.add(
                    f"ParallelismConfig.{parallelism}_handler is set, but {parallelism}_size is set to 1. This handler will be ignored."
                )

        if _warnings and accelerator.is_main_process:
            warnings.warn(
                "ParallelismConfig has the following warnings:\n" + "\n".join(_warnings),
                UserWarning,
            )
