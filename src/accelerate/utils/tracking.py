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
# limitations under the License.

import json
from typing import Any, Dict


def sanitize_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize configuration values to ensure they are compatible with tracking systems.
    
    This function converts complex data types (like lists, dicts) to JSON strings
    so they can be stored as hyperparameters in tracking systems that only support
    basic types (int, float, str, bool).
    
    Args:
        config (Dict[str, Any]): Configuration dictionary with potentially complex values
        
    Returns:
        Dict[str, Any]: Configuration dictionary with sanitized values
    """
    sanitized_config = {}
    
    for key, value in config.items():
        # Check if the value is of a supported type
        if isinstance(value, (int, float, str, bool, type(None))):
            sanitized_config[key] = value
        # Convert complex types to JSON string representation
        else:
            try:
                sanitized_config[key] = json.dumps(value)
            except (TypeError, ValueError):
                # If JSON serialization fails, convert to string
                sanitized_config[key] = str(value)
                
    return sanitized_config