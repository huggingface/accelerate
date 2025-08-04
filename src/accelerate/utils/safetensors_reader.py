# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.import io
import io
import json
import os
import struct
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem
from torch.futures import Future
from tqdm import tqdm


DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> tuple[Any, int]:
    # Copied from
    NUM_BYTES_FOR_HEADER_LEN = 8

    header_len_bytes = file_bytes.read(NUM_BYTES_FOR_HEADER_LEN)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return (metadata, header_len + NUM_BYTES_FOR_HEADER_LEN)


class SafetensorsReader(FileSystemReader):
    def __init__(self, path):
        super().__init__(path)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner):
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in tqdm(per_file.items()):
            new_path = self.fs.concat_path(self.path, relative_path)
            file_pointer = safe_open(new_path, framework="pt", device="cpu")
            for req in reqs:
                item_md = self.storage_data[req.storage_index]
                param = file_pointer.get_slice(req.storage_index.fqn)

                param = param[...]
                tensor = narrow_tensor_by_index(param, req.storage_offsets, req.lengths)
                target_tensor = planner.resolve_tensor(req).detach()

                assert target_tensor.size() == tensor.size(), (
                    f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def create_default_local_load_plan(
        state_dict: dict[str, Any], metadata: Metadata, strict: bool = True
    ) -> LoadPlan:
        return super().create_default_local_load_plan(
            state_dict=state_dict,
            metadata=metadata,
            strict=False,
        )

    def read_metadata(self) -> Metadata:
        meta = {}
        storage_data = {}
        for file in os.listdir(self.path):
            if file.endswith(".safetensors"):
                with self.fs.create_stream(self.fs.concat_path(self.path, file), "rb") as f:
                    metadata, metadata_size = _get_safetensors_file_metadata(f)

                    for key, value in metadata.items():
                        if key == "__metadata__":
                            continue
                        if key not in meta:
                            md = TensorStorageMetadata(
                                properties=TensorProperties(dtype=DTYPE_MAP[value["dtype"]]),
                                size=torch.Size(value["shape"]),
                                chunks=[
                                    ChunkStorageMetadata(
                                        offsets=torch.Size([0] * len(value["shape"])),
                                        sizes=torch.Size(value["shape"]),
                                    )
                                ],
                            )
                            meta[key] = md

                        else:
                            meta[key].chunks.append(
                                ChunkStorageMetadata(
                                    offsets=torch.Size([0] * len(value["shape"])),
                                    sizes=torch.Size(value["shape"]),
                                )
                            )

                        meta[key] = md
                        metadata_index = MetadataIndex(fqn=key, offset=[0] * len(value["shape"]))
                        storage_data[metadata_index] = _StorageInfo(
                            relative_path=file,
                            offset=value["data_offsets"][0] + metadata_size,
                            length=value["data_offsets"][1] - value["data_offsets"][0],
                        )

        metadata = Metadata(state_dict_metadata=meta, storage_data=storage_data)

        return metadata
