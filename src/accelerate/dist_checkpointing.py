# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License.import queue

import dataclasses
import os
import pickle
import queue
from io import UnsupportedOperation
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcs
from torch.distributed.checkpoint.filesystem import (
    FileSystemWriter,
    SavePlan,
    SavePlanner,
    _generate_uuid,
    _split_by_size_and_type,
)
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.storage import WriteResult


if TYPE_CHECKING:
    from accelerate import Accelerator


class AccelerateStorageWriter(FileSystemWriter):
    _DEFAULT_SUFFIX = ".distcp"
    _OPTIM_FILE_PATH = "optimizer_0"
    _MODEL_FILE_PATH = "pytorch_model_fsdp_0"

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.optim_path = self.fs.concat_path(self.path, self._OPTIM_FILE_PATH)
        self.model_path = self.fs.concat_path(self.path, self._MODEL_FILE_PATH)
        self.fs.mkdir(self.optim_path)
        self.fs.mkdir(self.model_path)
        return super().prepare_local_plan(plan)

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ):
        storage_plan = plan.storage_data
        optim_file_count = 0
        model_file_count = 0

        def gen_file(is_optimizer: bool = False) -> str:
            nonlocal optim_file_count, model_file_count
            if is_optimizer:
                optim_file_count += 1
                return f"{storage_plan.prefix}{optim_file_count}{self._DEFAULT_SUFFIX}"
            else:
                model_file_count += 1
                return f"{storage_plan.prefix}{model_file_count}{self._DEFAULT_SUFFIX}"

        file_queue: queue.Queue = queue.Queue()

        for bucket in _split_by_size_and_type(1, plan.items):
            optim_states = [wi for wi in bucket if "optim" in wi.index.fqn]
            model_states = [wi for wi in bucket if "model" in wi.index.fqn]

            for state, path in zip([optim_states, model_states], [self.optim_path, self.model_path]):
                file_name = gen_file()
                path = self.fs.concat_path(path, file_name)
                file_queue.put((path, file_name, state))

        return self._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        try:
            metadata = dataclasses.replace(metadata, version="1.0.0")
        except TypeError:
            pass

        def _split_metadata(
            metadata: Metadata,
        ) -> tuple[Metadata, Metadata]:
            result = []
            for to_get in ["model", "optim"]:
                result.append(
                    Metadata(
                        state_dict_metadata={
                            k.removeprefix("state."): v for k, v in metadata.state_dict_metadata.items() if to_get in k
                        },
                        planner_data={
                            k.removeprefix("state."): tuple([x for x in v if x != "state"])
                            for k, v in metadata.planner_data.items()
                            if to_get in k
                        },
                    )
                )

            return tuple(result)

        model_metadata, optim_metadata = _split_metadata(metadata)
        model_storage_md, optim_storage_md = {}, {}
        for wr_list in results:
            for wr in wr_list:
                new_index = dataclasses.asdict(wr.index)
                new_index["fqn"] = new_index["fqn"].removeprefix("state.")
                wr = WriteResult(
                    index=MetadataIndex(**new_index),
                    size_in_bytes=wr.size_in_bytes,
                    storage_data=wr.storage_data,
                )
                if "optim" in wr.index.fqn:
                    optim_storage_md.update({wr.index: wr.storage_data})
                else:
                    model_storage_md.update({wr.index: wr.storage_data})

        model_metadata.storage_data = model_storage_md
        optim_metadata.storage_data = optim_storage_md

        model_metadata.storage_meta = StorageMeta(self.model_path, save_id=_generate_uuid())
        optim_metadata.storage_meta = StorageMeta(self.optim_path, save_id=_generate_uuid())

        tmp_optim_path = cast(Path, self.fs.concat_path(self.optim_path, ".metadata.tmp"))
        tmp_model_path = cast(Path, self.fs.concat_path(self.model_path, ".metadata.tmp"))

        for meta, tmp_path, final_path in zip(
            [model_metadata, optim_metadata],
            [tmp_model_path, tmp_optim_path],
            [self.model_path, self.optim_path],
        ):
            with self.fs.create_stream(tmp_path, "wb") as metadata_file:
                pickle.dump(meta, metadata_file)
                if self.sync_files:
                    try:
                        os.fsync(metadata_file.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()

            metadata_path = self.fs.concat_path(final_path, ".metadata")
            if self.fs.exists(metadata_path):
                self.fs.rm_file(metadata_path)

            self.fs.rename(tmp_path, metadata_path)


def save_model_and_optimizer(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    async_save: bool = False,
) -> None:
    # async_save = False
    if getattr(accelerator, "_async_save_handle", None) is not None:
        accelerator._async_save_handle.result()

    options = dcs.StateDictOptions()

    import time

    accelerator.print(f"{time.asctime()} - Preparing state dict...")
    model_sd, optimizer_sd = dcs.get_state_dict(model, optimizer, options=options)
    accelerator.print(f"{time.asctime()} - Prepared state dict...")

    accelerator.print(f"{time.asctime()} - Saving state dict...")
    stateful = {
        "model": model_sd,
        "optimizer": optimizer_sd,
    }

    save_fn = dcp.save if not async_save else dcp.async_save

    potential_handle = dcp.async_save(
        state_dict=stateful,
        storage_writer=AccelerateStorageWriter(save_path),
    )
    accelerator.print(f"{time.asctime()} - Finished saving state dict...")

    if async_save:
        accelerator._async_save_handle = potential_handle
