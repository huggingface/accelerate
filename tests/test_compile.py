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
import unittest

import torch
from torch.utils.benchmark import Timer

from accelerate.test_utils import require_huggingface_suite, require_non_cpu, torch_device
from accelerate.utils import compile_regions, extract_model_from_parallel, release_memory


MODEL_ID = "gpt2"

COMPILE_ITERS = 2
INFERENCE_ITERS = 100

INFRENCE_STMT = "model(input_ids, use_cache=False)"
COMPILE_STMT = f"torch._dynamo.reset(); torch._inductor.utils.clear_inductor_caches(); {INFRENCE_STMT}"


@require_huggingface_suite
class RegionalCompilationTester(unittest.TestCase):
    def _get_model_and_inputs(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        with torch.device(torch_device):
            config = AutoConfig.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_config(config)
            input_ids = torch.randint(0, 1000, (4, 128), dtype=torch.int64)

        return model, input_ids

    def test_regions_are_compiled(self):
        model, _ = self._get_model_and_inputs()
        compiled_model = compile_regions(model, mode="reduce-overhead")

        # Check that the compiled model keeps a reference to the original model
        assert hasattr(compiled_model, "_orig_mod")
        assert compiled_model._orig_mod is model

        # Check that the compiled_model.transformer.h[i] and compiled_model.lm_head are compiled separately
        assert isinstance(compiled_model.transformer.h[0], torch._dynamo.eval_frame.OptimizedModule)
        assert isinstance(compiled_model.lm_head, torch._dynamo.eval_frame.OptimizedModule)

    def test_extract_model_keep_torch_compile(self):
        model, _ = self._get_model_and_inputs()
        compiled_model = compile_regions(model)

        distributed_model = torch.nn.parallel.DataParallel(model)
        distributed_compiled_model = compile_regions(distributed_model)
        compiled_model_unwrapped = extract_model_from_parallel(distributed_compiled_model, keep_torch_compile=True)

        assert compiled_model._orig_mod is compiled_model_unwrapped._orig_mod

    def test_extract_model_remove_torch_compile(self):
        model, _ = self._get_model_and_inputs()
        compiled_model = compile_regions(model)

        distributed_model = torch.nn.parallel.DataParallel(model)
        distributed_compiled_model = compile_regions(distributed_model)
        compiled_model_unwrapped = extract_model_from_parallel(distributed_compiled_model, keep_torch_compile=False)

        assert compiled_model._orig_mod is compiled_model_unwrapped

    @require_non_cpu
    @require_huggingface_suite
    def test_regional_compilation_cold_start(self):
        model, input_ids = self._get_model_and_inputs()

        regional_compilation_model = compile_regions(model)
        regional_compilation_cold_start = (
            Timer(stmt=COMPILE_STMT, globals={"model": regional_compilation_model, "input_ids": input_ids})
            .timeit(COMPILE_ITERS)
            .median
        )

        full_compilation_model = torch.compile(model)
        full_compilation_cold_start = (
            Timer(stmt=COMPILE_STMT, globals={"model": full_compilation_model, "input_ids": input_ids})
            .timeit(COMPILE_ITERS)
            .median
        )

        self.assertLess(
            regional_compilation_cold_start,
            full_compilation_cold_start,
            "Regional compilation should have a faster cold start than full compilation",
        )

        release_memory(model, full_compilation_model, regional_compilation_model)

    @require_non_cpu
    @require_huggingface_suite
    def test_regional_compilation_inference_speedup(self):
        model, input_ids = self._get_model_and_inputs()

        baseline_inference_latency = (
            Timer(stmt=INFRENCE_STMT, globals={"model": model, "input_ids": input_ids}).timeit(INFERENCE_ITERS).median
        )

        regional_compilation_model = compile_regions(model)
        regional_compilation_inference_latency = (
            Timer(stmt=INFRENCE_STMT, globals={"model": regional_compilation_model, "input_ids": input_ids})
            .timeit(INFERENCE_ITERS)
            .median
        )

        full_compilation_model = torch.compile(model)
        full_compilation_inference_latency = (
            Timer(stmt=INFRENCE_STMT, globals={"model": full_compilation_model, "input_ids": input_ids})
            .timeit(INFERENCE_ITERS)
            .median
        )

        full_compilation_inference_speedup = baseline_inference_latency / full_compilation_inference_latency
        regional_compilation_inference_speedup = baseline_inference_latency / regional_compilation_inference_latency

        self.assertAlmostEqual(
            regional_compilation_inference_speedup,
            full_compilation_inference_speedup,
            delta=0.1,
            msg="Regional compilation should have a similar speedup to full compilation",
        )

        release_memory(model, full_compilation_model, regional_compilation_model)
