# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.utils.dataclasses import DynamoBackend


def test_dynamo_backend_config_options_match_enum():
    for backend in DYNAMO_BACKENDS:
        assert DynamoBackend(backend).value == backend
