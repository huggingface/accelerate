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

import torch
from torch.utils.benchmark import Compare, Timer
from transformers import AutoConfig, AutoModelForCausalLM

from accelerate.test_utils.testing import get_backend
from accelerate.utils import compile_regions


torch.set_float32_matmul_precision("high")

COMPILE_ITERS = 2
INFERENCE_ITERS = 100

BASELINE = "Baseline"
COMPILE_TIME = "Compile time"
INFRENCE_TIME = "Inference time"
FULL_COMPILATION = "Full compilation"
REGIONAL_COMPILATION = "Regional compilation"

INFRENCE_STMT = "model(input_ids, use_cache=False)"
COMPILE_STMT = f"torch._dynamo.reset(); torch._inductor.utils.clear_inductor_caches(); {INFRENCE_STMT}"

torch_device_type, _, _ = get_backend()

results = []
for model_id in [
    # non-gated llama models
    "NousResearch/Llama-3.2-1B",
    "NousResearch/Hermes-3-Llama-3.2-3B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "NousResearch/Nous-Hermes-Llama2-13b",
]:
    with torch.device(torch_device_type):
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float16).eval()

    full_compilation_model = torch.compile(model)
    regional_compilation_model = compile_regions(model)

    for model, sub_label, description, stmt, iters in [
        (model, BASELINE, INFRENCE_TIME, INFRENCE_STMT, INFERENCE_ITERS),
        (full_compilation_model, FULL_COMPILATION, COMPILE_TIME, COMPILE_STMT, COMPILE_ITERS),
        (full_compilation_model, FULL_COMPILATION, INFRENCE_TIME, INFRENCE_STMT, INFERENCE_ITERS),
        (regional_compilation_model, REGIONAL_COMPILATION, COMPILE_TIME, COMPILE_STMT, COMPILE_ITERS),
        (regional_compilation_model, REGIONAL_COMPILATION, INFRENCE_TIME, INFRENCE_STMT, INFERENCE_ITERS),
    ]:
        for batch_size, sequence_length in [(1, 128), (4, 128)]:
            input_ids = torch.randint(
                0, 1000, size=(batch_size, sequence_length), dtype=torch.int64, device=torch_device_type
            )
            results.append(
                Timer(
                    label=model_id,
                    sub_label=sub_label,
                    description=f"{description} ({batch_size}x{sequence_length})",
                    globals={"model": model, "input_ids": input_ids},
                    stmt=stmt,
                ).timeit(number=iters)
            )

compare = Compare(results)
compare.colorize()
compare.print()
