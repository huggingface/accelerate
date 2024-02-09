# coding=utf-8
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
import time

import torch
from transformers import AutoModelForMaskedLM

from accelerate import PartialState, prepare_pippy
from accelerate.utils import set_seed


# Set the random seed to have reproducable outputs
set_seed(42)

# Create an example model
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

# Input configs
# Create example inputs for the model
input = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(2, 512),  # bs x seq_len
    device="cpu",
    dtype=torch.int64,
    requires_grad=False,
)


# Create a pipeline stage from the model
# Using `auto` is equivalent to letting `device_map="auto"` figure
# out device mapping and will also split the model according to the
# number of total GPUs available if it fits on one GPU
model = prepare_pippy(model, split_points="auto", example_args=(input,))

# You can pass `gather_output=True` to have the output from the model
# available on all GPUs
# model = prepare_pippy(model, split_points="auto", example_args=(input,), gather_output=True)

# Move the inputs to the first device
input = input.to("cuda:0")

# Take an average of 5 times
# Measure first batch
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    output = model(input)
torch.cuda.synchronize()
end_time = time.time()
first_batch = end_time - start_time

# Now that CUDA is init, measure after
torch.cuda.synchronize()
start_time = time.time()
for i in range(5):
    with torch.no_grad():
        output = model(input)
torch.cuda.synchronize()
end_time = time.time()

# The outputs are only on the final process by default
if PartialState().is_last_process:
    output = torch.stack(tuple(output[0]))
    print(f"Time of first pass: {first_batch}")
    print(f"Average time per batch: {(end_time - start_time)/5}")
