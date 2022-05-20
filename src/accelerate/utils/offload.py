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

import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union

import numpy as np
import torch


def offload_state_dict(save_dir: Union[str, os.PathLike], state_dict: Dict[str, torch.Tensor]):
    """
    Offload a state dict in a given folder.

    Args:
        save_dir (`str` or `os.PathLike`): The directory in which to offload the state dict.
        state_dict (`Dict[str, torch.Tensor]`): The dictionary of tensors to offload.
    """
    os.makedirs(save_dir, exist_ok=True)
    index = {}
    for name, parameter in state_dict.items():
        tensor_file = os.path.join(save_dir, f"{name}.dat")
        array = parameter.numpy()
        index[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
        if array.ndim == 0:
            array = array[None]
        file_array = np.memmap(tensor_file, dtype=array.dtype, mode="w+", shape=array.shape)
        file_array[:] = array[:]
        file_array.flush()

    # Update index
    index_file = os.path.join(save_dir, "index.json")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            current_index = json.load(f)
    else:
        current_index = {}
    current_index.update(index)

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(current_index, f, indent=2)


def offload_weight(weight, weight_name, offload_folder, index=None):
    array = weight.numpy()
    tensor_file = os.path.join(offload_folder, f"{weight_name}.dat")
    if index is not None:
        index[weight_name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
    file_array = np.memmap(tensor_file, dtype=array.dtype, mode="w+", shape=array.shape)
    file_array[:] = array[:]
    file_array.flush()
    return index


def save_offload_index(index, offload_folder):
    if index is None or len(index) == 0:
        # Nothing to save
        return

    offload_index_file = os.path.join(offload_folder, "index.json")
    if os.path.isfile(offload_index_file):
        with open(offload_index_file, "r", encoding="utf-8") as f:
            current_index = json.load(f)
    else:
        current_index = {}
    current_index.update(index)

    with open(offload_index_file, "w", encoding="utf-8") as f:
        json.dump(current_index, f, indent=2)


class PrefixedDataset(Mapping):
    """
    Will access keys in a given dataset by adding a prefix.

    Args:
        dataset (`Mapping`): Any map with string keys.
        prefix (`str`): A prefix to add when trying to access any element in the underlying dataset.
    """

    def __init__(self, dataset: Mapping, prefix: str):
        self.dataset = dataset
        self.prefix = prefix

    def __getitem__(self, key):
        return self.dataset[f"{self.prefix}{key}"]

    def __iter__(self):
        return iter([key for key in self.dataset if key.startswith(self.prefix)])

    def __len__(self):
        return len(self.dataset)


class OffloadedWeightsLoader(Mapping):
    """
    A collection that loads weights stored in a given state dict or memory-mapped on disk.

    Args:
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            A dictionary parameter name to tensor.
        save_folder (`str` or `os.PathLike`, *optional*):
            The directory in which the weights are stored (by `offload_state_dict` for instance).
        index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype` and `shape`). Will default to the index saved
            in `save_folder`.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor] = None,
        save_folder: Optional[Union[str, os.PathLike]] = None,
        index: Mapping = None,
    ):
        if state_dict is None and save_folder is None:
            raise ValueError("Need either a `state_dict` or a `save_folder` containing offloaded weights.")

        self.state_dict = {} if state_dict is None else state_dict
        self.save_folder = save_folder
        if index is None and save_folder is not None:
            with open(os.path.join(save_folder, "index.json")) as f:
                index = json.load(f)
        self.index = {} if index is None else index
        self.all_keys = list(self.state_dict.keys())
        self.all_keys.extend([key for key in self.index if key not in self.all_keys])

    def __getitem__(self, key: str):
        # State dict gets priority
        if key in self.state_dict:
            return self.state_dict[key]
        weight_info = self.index[key]
        weight_file = os.path.join(self.save_folder, f"{key}.dat")
        shape = tuple(weight_info["shape"])
        if shape == ():
            weight = np.memmap(weight_file, dtype=weight_info["dtype"], shape=(1,), mode="r")[0]
        else:
            weight = np.memmap(weight_file, dtype=weight_info["dtype"], shape=shape, mode="r")
        return torch.tensor(weight)

    def __iter__(self):
        return iter(self.all_keys)

    def __len__(self):
        return len(self.all_keys)


def extract_submodules_state_dict(state_dict: Dict[str, torch.Tensor], submodule_names: List[str]):
    """
    Extract the sub state-dict corresponding to a list of given submodules.

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dict to extract from.
        submodule_names (`List[str]`): The list of submodule names we want to extract.
    """
    result = {}
    for module_name in submodule_names:
        result.update({key: param for key, param in state_dict.items() if key.startswith(module_name)})
    return result
