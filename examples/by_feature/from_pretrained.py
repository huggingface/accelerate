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
# limitations under the License.
import os
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from transformers.models.llama4 import Llama4ForCausalLM, Llama4TextConfig

from accelerate import Accelerator
from accelerate.utils import ParallelismConfig, SafetensorsReader


def test_read():
    pc = ParallelismConfig(tp_size=int(os.environ["WORLD_SIZE"]))

    accelerator = Accelerator(parallelism_config=pc)

    with torch.device("meta"):
        config = Llama4TextConfig.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
        model = Llama4ForCausalLM.from_config(config)

        model = accelerator.prepare(model)

    model = model.to_empty(device="cuda")

    sd = model.state_dict()

    start_time = time.perf_counter()

    dcp.load(
        state_dict=sd,
        storage_reader=SafetensorsReader("models/Llama4"),
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    end_time = time.perf_counter()
    accelerator.print(f"Time taken to read safetensors: {end_time - start_time:.2f} seconds")

    set_model_state_dict(model, sd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = accelerator.prepare(optimizer)

    inputs = torch.randint(0, 1000, (1, 128), device="cuda")
    outputs = model(inputs, labels=inputs)

    from torch.distributed.tensor.experimental import implicit_replication

    with implicit_replication():
        accelerator.backward(outputs.loss)
    optimizer.step()

    return


if __name__ == "__main__":
    test_read()
