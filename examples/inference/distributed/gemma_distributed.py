# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Distributed inference example using Google Gemma 2B with Accelerate.

This example demonstrates how to run distributed inference on multiple GPUs
using the `PartialState` API for simple data-parallel inference.

Run with:
    accelerate launch --num_processes {NUM_GPUS} gemma_distributed.py

Or:
    torchrun --nproc-per-node {NUM_GPUS} gemma_distributed.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState
from accelerate.utils import gather_object


# Start up the distributed environment without needing the Accelerator.
distributed_state = PartialState()

# Use Google Gemma 2B - a small but capable model good for demo purposes.
model_name = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=distributed_state.device, torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Example prompts covering different tasks
prompts = [
    "Explain quantum computing in simple terms:",
    "Write a Python function to compute fibonacci numbers:",
    "Translate to French: The weather is nice today.",
    "Summarize the benefits of distributed computing:",
    "What is the capital of Australia?",
]

# Configure batching - adjust based on your GPU memory
batch_size = 2
pad_to_multiple_of = 8

# Split prompts into batches
formatted_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

# Apply padding on the left for generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"
tokenized_prompts = [
    tokenizer(
        formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt"
    )
    for formatted_prompt in formatted_prompts
]
tokenizer.padding_side = padding_side_default

completions_per_process = []

# Distribute batches across processes
# Each process gets a subset of batches; apply_padding ensures equal batch counts
with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    for batch in batched_prompts:
        batch = batch.to(distributed_state.device)
        # Generate with temperature for more natural responses
        outputs = model.generate(
            **batch, max_new_tokens=50, temperature=0.7, do_sample=True, top_p=0.9
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions_per_process.extend(generated_text)

# Gather results from all processes
completions_gather = gather_object(completions_per_process)

# Remove duplicates from apply_padding
completions = completions_gather[: len(prompts)]

# Print results from main process
distributed_state.print("=== Distributed Inference Results ===")
for i, (prompt, completion) in enumerate(zip(prompts, completions)):
    distributed_state.print(f"\n--- Prompt {i+1} ---")
    distributed_state.print(f"Prompt: {prompt}")
    distributed_state.print(f"Completion: {completion}")

distributed_state.print("\n✅ Distributed inference completed successfully!")
