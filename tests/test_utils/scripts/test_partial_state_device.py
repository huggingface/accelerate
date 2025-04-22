# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""
Testing script for proper device initialization in PartialState
to prevent duplicate GPU usage in distributed processing.
"""

import argparse
import torch
from accelerate.state import PartialState
from accelerate.utils import gather_object

def test_padding_behavior(distributed_state):
    """
    Tests the padding behavior when there are extra samples to distribute.
    This test verifies the fix from PR #3518 where padding calculation was modified
    to properly handle extra samples.
    """
    # Create test data with 5 samples to be distributed across 3 processes
    # This should result in 2 processes getting 2 samples and 1 process getting 1 sample
    test_data = [f"sample_{i}" for i in range(5)]
    
    # Process the data with padding
    processed_data = []
    with distributed_state.split_between_processes(test_data, apply_padding=True) as batched_data:
        processed_data.extend(batched_data)
    
    # Gather results from all processes
    gathered_data = gather_object(processed_data)
    
    if distributed_state.is_main_process:
        print("Gathered data:", gathered_data)
        
        # Verify padding behavior
        # With 5 samples and 3 processes:
        # - First process should get 2 samples + padding
        # - Second process should get 2 samples + padding
        # - Third process should get 1 sample + padding
        expected_lengths = [3, 3, 3]  # Each process should have 3 elements after padding
        
        # Check that each process's data has the correct length
        for i, data in enumerate(gathered_data):
            assert len(data) == expected_lengths[i], f"Process {i} has incorrect length: {len(data)} != {expected_lengths[i]}"
            
            # Check that padding uses the last element
            if len(data) > 2:  # For processes that needed padding
                assert data[-1] == data[-2], f"Process {i} padding not using last element: {data[-1]} != {data[-2]}"
        
        print("Padding test passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-padding", action="store_true", help="Run the padding test")
    args = parser.parse_args()

    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    # Initialize distributed state
    distributed_state = PartialState()
    
    # Print process and device information
    print(f"Process index: {distributed_state.process_index}")
    print(f"Local process index: {distributed_state.local_process_index}")
    print(f"Device: {distributed_state.device}")
    
    if args.test_padding:
        test_padding_behavior(distributed_state)
    else:
        # Original test code
        prompts = [str(i) for i in range(4)]  # ["0", "1", "2", "3"]
        batch_size = 2
        
        # Split into batches
        tokenized_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
        
        # Process the data
        completions_per_process = []
        with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
            for batch in batched_prompts:
                # Generate text that includes the device information
                generated_text = [f"{distributed_state.device}: {t}" for t in batch]
                completions_per_process.extend(generated_text)
        
        # Gather results from all processes
        completions_gathered = gather_object(completions_per_process)
        
        # Check for duplicate device usage
        if distributed_state.is_main_process:
            print("Gathered completions:", completions_gathered)
            
            # Check if the same device appears multiple times when we have more processes than devices
            if torch.cuda.is_available():
                devices_used = set()
                duplicate_found = False
                
                for completion in completions_gathered:
                    device_str = completion.split(':')[0]
                    if "cuda" in device_str and device_str in devices_used:
                        duplicate_found = True
                        break
                    devices_used.add(device_str)
                
                if not duplicate_found:
                    print("No duplicate device usage detected - test passed!")
                else:
                    print("WARNING: Duplicate device usage detected!")
    
    # Synchronize at the end
    distributed_state.wait_for_everyone()
