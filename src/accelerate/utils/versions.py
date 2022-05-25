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

import sys
from typing import Union

from packaging.version import Version, parse

from .constants import STR_OPERATION_TO_FUNC


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

torch_version = parse(importlib_metadata.version("torch"))


def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """Compares `library_or_version` to `requirement_version` with an `operation`

    Args:
        library_or_version (`str`, `packaging.version.Version]`):
            A library name or Version to check
        operation (`str`):
            A string representation of an operator, such as ">" or "<="
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def check_torch_version(operation: str, version: str):
    """Compares the current PyTorch version to `version` with an `operation`

    Args:
        operation (`str`):
            A string representation of an operator, such as ">" or "<="
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(torch_version, operation, version)
