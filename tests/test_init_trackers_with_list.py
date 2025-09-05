#!/usr/bin/env python3
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

import argparse
import os
import sys
import tempfile
import traceback

# Add the local accelerate directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the sanitize function directly from the file
from accelerate.utils.tracking import sanitize_config_values
from accelerate import Accelerator


def test_sanitize_function():
    """Test our sanitize_config_values function directly"""
    # Create test data with list values
    test_config = {
        "int_value": 42,
        "float_value": 3.14,
        "str_value": "hello",
        "bool_value": True,
        "none_value": None,
        "list_value": ["item1", "item2", "item3"],
        "dict_value": {"key": "value"},
        "nested_list": [["nested", "list"], ["another", "one"]]
    }
    
    # Sanitize the config
    sanitized = sanitize_config_values(test_config)
    
    # Check that all values are now basic types
    for key, value in sanitized.items():
        print(f"{key}: {value} (type: {type(value)})")
        if not isinstance(value, (int, float, str, bool, type(None))):
            print(f"ERROR: Value for {key} is not a basic type!")
            return False
    
    print("SUCCESS: sanitize_config_values works correctly")
    return True


def test_init_trackers_with_list():
    """Test that init_trackers works with list values in config."""
    # Create an argument parser with a list argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=["A photo of sks dog in a bucket", "A sks cat wearing a coat"],
        nargs="*",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    
    # Parse the arguments
    args = parser.parse_args([])
    
    # Show what the args look like
    print("Original args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    # Show what happens after sanitization
    sanitized = sanitize_config_values(vars(args))
    print("\nSanitized args:")
    for key, value in sanitized.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    # Create a temporary directory for logging
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create accelerator with tensorboard logging
        accelerator = Accelerator(log_with=["tensorboard"], project_dir=tmpdir)
        
        # Initialize trackers with config containing list values
        # This should not raise an error
        try:
            accelerator.init_trackers("test_project", config=vars(args))
            print("\nSUCCESS: init_trackers worked with list values in config")
            accelerator.end_training()
        except Exception as e:
            print(f"\nFAILED: init_trackers failed with error: {e}")
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    print("Testing sanitize_config_values function:")
    success1 = test_sanitize_function()
    
    print("\nTesting init_trackers with list values:")
    success2 = test_init_trackers_with_list()
    
    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)