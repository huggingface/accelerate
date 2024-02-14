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

import gc
import tempfile
import unittest

import torch
import torch.nn as nn

from accelerate import Accelerator, init_empty_weights
from accelerate.test_utils import (
    require_bnb,
    require_cuda,
    require_huggingface_suite,
    require_multi_gpu,
    require_non_torch_xla,
    slow,
)
from accelerate.utils.bnb import load_and_quantize_model
from accelerate.utils.dataclasses import BnbQuantizationConfig


class BitsAndBytesConfigIntegration(unittest.TestCase):
    def test_BnbQuantizationConfig(self):
        with self.assertRaises(ValueError):
            BnbQuantizationConfig(load_in_8bit=True, load_in_4bit=True)


@require_non_torch_xla
@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class MixedInt8EmptyModelTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "marcsun13/bloom-1b7_with_lm_head"

    # Constant values
    # This was obtained on a Quadro RTX 8000 so the number might slightly change
    EXPECTED_RELATIVE_DIFFERENCE = 1.540025

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        """
        Setup quantized model from empty model
        """
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

        # create model on meta device
        with init_empty_weights():
            self.model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        self.model_8bit.tie_weights()

        self.weights_location = hf_hub_download(self.model_name, "pytorch_model.bin")
        self.bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        self.model_8bit = load_and_quantize_model(
            self.model_8bit,
            self.bnb_quantization_config,
            weights_location=self.weights_location,
            device_map={"": 0},
            no_split_module_classes=["BloomBlock"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
        self.accelerate = Accelerator()

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Int8Params

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_8bit = self.model_8bit.get_memory_footprint()

        assert round((mem_fp16 / mem_8bit) - self.EXPECTED_RELATIVE_DIFFERENCE, 7) >= 0
        assert self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params

    def test_linear_are_8bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_not_converted = (
                    self.bnb_quantization_config.keep_in_fp32_modules + self.bnb_quantization_config.skip_modules
                )
                if name not in modules_not_converted:
                    assert module.weight.dtype == torch.int8

    def test_llm_skip(self):
        r"""
        A simple test to check if `llm_int8_skip_modules` works as expected
        """
        import bitsandbytes as bnb
        from transformers import AutoConfig, AutoModelForCausalLM

        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, skip_modules=["lm_head", "transformer.word_embeddings"]
        )

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model.tie_weights()
        model = load_and_quantize_model(
            model,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map="auto",
            no_split_module_classes=["BloomBlock"],
        )

        assert model.transformer.h[1].mlp.dense_4h_to_h.weight.dtype == torch.int8
        assert isinstance(model.transformer.h[1].mlp.dense_4h_to_h, bnb.nn.Linear8bitLt)
        assert isinstance(model.lm_head, nn.Linear)
        assert model.lm_head.weight.dtype != torch.int8

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_parallel = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        output_text = self.tokenizer.decode(output_parallel[0], skip_special_tokens=True)
        assert output_text == self.EXPECTED_OUTPUT

    def test_generate_quality(self):
        self.check_inference_correctness(self.model_8bit)

    def test_fp32_8bit_conversion(self):
        r"""
        Test whether it is possible to mix both `8bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, keep_in_fp32_modules=["lm_head"])

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model.tie_weights()
        model = load_and_quantize_model(
            model,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map="auto",
            no_split_module_classes=["BloomBlock"],
        )
        assert model.lm_head.weight.dtype == torch.float32

    @require_multi_gpu
    def test_cpu_gpu_loading_custom_device_map(self):
        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": 0,
            "lm_head": "cpu",
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": 0,
            "transformer.h.4": 0,
            "transformer.h.5": 0,
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 1,
            "transformer.h.10": 0,
            "transformer.h.11": 1,
            "transformer.h.12": 0,
            "transformer.h.13": 0,
            "transformer.h.14": 1,
            "transformer.h.15": 0,
            "transformer.h.16": 0,
            "transformer.h.17": 1,
            "transformer.h.18": 1,
            "transformer.h.19": 0,
            "transformer.h.20": 1,
            "transformer.h.21": 1,
            "transformer.h.22": 0,
            "transformer.h.23": 0,
            "transformer.ln_f": 1,
        }
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        with init_empty_weights():
            model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model_8bit.tie_weights()
        model_8bit = load_and_quantize_model(
            model_8bit,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map=device_map,
            no_split_module_classes=["BloomBlock"],
        )
        assert model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params
        assert model_8bit.transformer.h[1].mlp.dense_4h_to_h.weight.__class__ == Int8Params
        self.check_inference_correctness(model_8bit)

    @require_multi_gpu
    def test_cpu_gpu_loading_custom_device_map_offload_state_dict(self):
        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map` and offload_state_dict=True.
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": 0,
            "lm_head": "cpu",
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": 0,
            "transformer.h.4": 0,
            "transformer.h.5": 0,
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 1,
            "transformer.h.10": 0,
            "transformer.h.11": 1,
            "transformer.h.12": 0,
            "transformer.h.13": 0,
            "transformer.h.14": 1,
            "transformer.h.15": 0,
            "transformer.h.16": 0,
            "transformer.h.17": 1,
            "transformer.h.18": 1,
            "transformer.h.19": 0,
            "transformer.h.20": 1,
            "transformer.h.21": 1,
            "transformer.h.22": 0,
            "transformer.h.23": 0,
            "transformer.ln_f": 1,
        }

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        with init_empty_weights():
            model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model_8bit.tie_weights()
        model_8bit = load_and_quantize_model(
            model_8bit,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map=device_map,
            no_split_module_classes=["BloomBlock"],
            offload_state_dict=True,
        )
        assert model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params
        assert model_8bit.transformer.h[1].mlp.dense_4h_to_h.weight.__class__ == Int8Params
        self.check_inference_correctness(model_8bit)

    @require_multi_gpu
    def test_cpu_gpu_disk_loading_custom_device_map_kwargs(self):
        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        This time we also add `disk` on the device_map - using the kwargs directly instead of the quantization config
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": 0,
            "lm_head": "cpu",
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": "disk",
            "transformer.h.4": "disk",
            "transformer.h.5": "disk",
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 1,
            "transformer.h.10": 0,
            "transformer.h.11": 1,
            "transformer.h.12": 0,
            "transformer.h.13": 0,
            "transformer.h.14": 1,
            "transformer.h.15": 0,
            "transformer.h.16": 0,
            "transformer.h.17": 1,
            "transformer.h.18": 1,
            "transformer.h.19": 0,
            "transformer.h.20": 1,
            "transformer.h.21": 1,
            "transformer.h.22": 0,
            "transformer.h.23": 0,
            "transformer.ln_f": 1,
        }
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        with init_empty_weights():
            model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        model_8bit.tie_weights()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_8bit = load_and_quantize_model(
                model_8bit,
                bnb_quantization_config,
                weights_location=self.weights_location,
                device_map=device_map,
                no_split_module_classes=["BloomBlock"],
                offload_folder=tmpdirname,
                offload_state_dict=True,
            )
            assert model_8bit.transformer.h[4].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            assert model_8bit.transformer.h[5].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            self.check_inference_correctness(model_8bit)

    def test_int8_serialization(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit.
        """
        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmpdirname:
            # saving state dict for now but will save config and other in the future
            self.accelerate.save_model(self.model_8bit, tmpdirname)

            with init_empty_weights():
                # let's suppose that we can get the right config
                model_8bit_from_saved = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
            model_8bit_from_saved.tie_weights()

            bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

            model_8bit_from_saved = load_and_quantize_model(
                model_8bit_from_saved,
                bnb_quantization_config,
                weights_location=tmpdirname,
                device_map="auto",
                no_split_module_classes=["BloomBlock"],
            )

            assert model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            assert hasattr(model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "SCB")
            assert hasattr(model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "CB")

            self.check_inference_correctness(model_8bit_from_saved)

    @require_multi_gpu
    def test_int8_serialization_offload(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit and offload weights to cpu/disk
        """

        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmpdirname:
            # saving state dict for now but will save config and other in the future
            self.accelerate.save_model(self.model_8bit, tmpdirname)

            with init_empty_weights():
                # let's suppose that we can get the right config
                model_8bit_from_saved = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
            model_8bit_from_saved.tie_weights()
            bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)
            device_map = {
                "transformer.word_embeddings": "cpu",
                "transformer.word_embeddings_layernorm": 0,
                "lm_head": "cpu",
                "transformer.h.0": "cpu",
                "transformer.h.1": "cpu",
                "transformer.h.2": "cpu",
                "transformer.h.3": "disk",
                "transformer.h.4": "disk",
                "transformer.h.5": "disk",
                "transformer.h.6": 0,
                "transformer.h.7": 0,
                "transformer.h.8": 0,
                "transformer.h.9": 1,
                "transformer.h.10": 0,
                "transformer.h.11": 1,
                "transformer.h.12": 0,
                "transformer.h.13": 0,
                "transformer.h.14": 1,
                "transformer.h.15": 0,
                "transformer.h.16": 0,
                "transformer.h.17": 1,
                "transformer.h.18": 1,
                "transformer.h.19": 0,
                "transformer.h.20": 1,
                "transformer.h.21": 1,
                "transformer.h.22": 0,
                "transformer.h.23": 0,
                "transformer.ln_f": 1,
            }
            model_8bit_from_saved = load_and_quantize_model(
                model_8bit_from_saved,
                bnb_quantization_config,
                weights_location=tmpdirname,
                device_map=device_map,
                no_split_module_classes=["BloomBlock"],
                offload_folder=tmpdirname + "/tmp",
                offload_state_dict=True,
            )

            assert model_8bit_from_saved.transformer.h[4].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            assert model_8bit_from_saved.transformer.h[5].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            self.check_inference_correctness(model_8bit_from_saved)

    def test_int8_serialization_shard(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit.
        """
        from bitsandbytes.nn import Int8Params
        from transformers import AutoConfig, AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmpdirname:
            # saving state dict for now but will save config and other in the future
            self.accelerate.save_model(self.model_8bit, tmpdirname, max_shard_size="1GB")

            with init_empty_weights():
                # let's suppose that we can get the right config
                model_8bit_from_saved = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

            model_8bit_from_saved.tie_weights()

            bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

            model_8bit_from_saved = load_and_quantize_model(
                model_8bit_from_saved,
                bnb_quantization_config,
                weights_location=tmpdirname,
                device_map="auto",
                no_split_module_classes=["BloomBlock"],
            )

            assert model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params
            assert hasattr(model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "SCB")
            assert hasattr(model_8bit_from_saved.transformer.h[0].mlp.dense_4h_to_h.weight, "CB")

            self.check_inference_correctness(model_8bit_from_saved)


@require_non_torch_xla
@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class MixedInt8LoaddedModelTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "marcsun13/bloom-1b7_with_lm_head"

    # Constant values
    # This was obtained on a Quadro RTX 8000 so the number might slightly change
    EXPECTED_RELATIVE_DIFFERENCE = 1.540025

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        """
        Setup quantized model from loaded model
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

        self.bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model_8bit = load_and_quantize_model(self.model_8bit, self.bnb_quantization_config)

        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Int8Params

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_8bit = self.model_8bit.get_memory_footprint()

        assert round((mem_fp16 / mem_8bit) - self.EXPECTED_RELATIVE_DIFFERENCE, 7) >= 0
        assert self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params

    def test_linear_are_8bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_not_converted = (
                    self.bnb_quantization_config.keep_in_fp32_modules + self.bnb_quantization_config.skip_modules
                )
                if name not in modules_not_converted:
                    assert module.weight.dtype == torch.int8

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        output_sequences = self.model_8bit.generate(
            input_ids=encoded_input["input_ids"].to(self.model_8bit.device), max_new_tokens=10
        )

        assert self.tokenizer.decode(output_sequences[0], skip_special_tokens=True) == self.EXPECTED_OUTPUT

    def test_fp32_8bit_conversion(self):
        r"""
        Test whether it is possible to mix both `8bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, keep_in_fp32_modules=["lm_head"])

        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        model = load_and_quantize_model(model, bnb_quantization_config)
        assert model.lm_head.weight.dtype == torch.float32


@require_non_torch_xla
@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class Bnb4BitEmptyModelTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "marcsun13/bloom-1b7_with_lm_head"

    # Constant values
    # This was obtained on a RTX Titan so the number might slightly change
    EXPECTED_RELATIVE_DIFFERENCE = 2.109659552692574

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John.\nI am a friend of your father.\n")
    MAX_NEW_TOKENS = 10

    def setUp(self):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

        # create model on meta device
        with init_empty_weights():
            self.model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        self.model_4bit.tie_weights()
        self.weights_location = hf_hub_download(self.model_name, "pytorch_model.bin")
        self.bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        self.model_4bit = load_and_quantize_model(
            self.model_4bit,
            self.bnb_quantization_config,
            weights_location=self.weights_location,
            device_map={"": 0},
            no_split_module_classes=["BloomBlock"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

    def tearDown(self):
        """
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        super().tearDown()
        del self.model_fp16
        del self.model_4bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Params4bit

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()

        assert round((mem_fp16 / mem_4bit) - self.EXPECTED_RELATIVE_DIFFERENCE, 7) >= 0
        assert self.model_4bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Params4bit

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        assert self.tokenizer.decode(output_sequences[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    def test_generate_quality(self):
        self.check_inference_correctness(self.model_4bit)

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if (
                    name
                    not in self.bnb_quantization_config.keep_in_fp32_modules
                    + self.bnb_quantization_config.skip_modules
                ):
                    # 4-bit parameters are packed in uint8 variables
                    assert module.weight.dtype == torch.uint8

    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, keep_in_fp32_modules=["lm_head"])

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model.tie_weights()
        model = load_and_quantize_model(
            model,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map="auto",
            no_split_module_classes=["BloomBlock"],
        )
        assert model.lm_head.weight.dtype == torch.float32

    @require_multi_gpu
    def test_cpu_gpu_loading_random_device_map(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a random `device_map`.
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": 0,
            "lm_head": "cpu",
            "transformer.h.0": 0,
            "transformer.h.1": 0,
            "transformer.h.2": 0,
            "transformer.h.3": 0,
            "transformer.h.4": 0,
            "transformer.h.5": 0,
            "transformer.h.6": 0,
            "transformer.h.7": 0,
            "transformer.h.8": 0,
            "transformer.h.9": 1,
            "transformer.h.10": 0,
            "transformer.h.11": 1,
            "transformer.h.12": 0,
            "transformer.h.13": 0,
            "transformer.h.14": 1,
            "transformer.h.15": 0,
            "transformer.h.16": 0,
            "transformer.h.17": 1,
            "transformer.h.18": 1,
            "transformer.h.19": 0,
            "transformer.h.20": 1,
            "transformer.h.21": 1,
            "transformer.h.22": 0,
            "transformer.h.23": 0,
            "transformer.ln_f": 1,
        }

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        with init_empty_weights():
            model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model_4bit.tie_weights()
        model_4bit = load_and_quantize_model(
            model_4bit,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map=device_map,
            no_split_module_classes=["BloomBlock"],
        )
        self.check_inference_correctness(model_4bit)

    @require_multi_gpu
    def test_cpu_gpu_loading_custom_device_map(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a random `device_map`.
        """
        device_map = {
            "transformer.word_embeddings": "cpu",
            "transformer.word_embeddings_layernorm": "cpu",
            "lm_head": "cpu",
            "transformer.h": 0,
            "transformer.ln_f": 1,
        }

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        with init_empty_weights():
            model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model_4bit.tie_weights()
        model_4bit = load_and_quantize_model(
            model_4bit,
            bnb_quantization_config,
            weights_location=self.weights_location,
            device_map=device_map,
            no_split_module_classes=["BloomBlock"],
        )
        self.check_inference_correctness(model_4bit)

    @require_multi_gpu
    def test_cpu_gpu_disk_loading_custom_device_map_kwargs(self):
        from transformers import AutoConfig, AutoModelForCausalLM

        r"""
        A test to check is dispatching a model on cpu & gpu works correctly using a custom `device_map`.
        This time we also add `disk` on the device_map - using the kwargs directly instead of the quantization config
        """
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.word_embeddings_layernorm": "disk",
            "lm_head": 0,
            "transformer.h": 1,
            "transformer.ln_f": "cpu",
        }
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        with init_empty_weights():
            model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        model_4bit.tie_weights()
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_4bit = load_and_quantize_model(
                model_4bit,
                bnb_quantization_config,
                weights_location=self.weights_location,
                device_map=device_map,
                no_split_module_classes=["BloomBlock"],
                offload_folder=tmpdirname,
                offload_state_dict=True,
            )
            self.check_inference_correctness(model_4bit)


@require_non_torch_xla
@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class Bnb4BitTestLoadedModel(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "marcsun13/bloom-1b7_with_lm_head"

    # Constant values
    # This was obtained on a RTX Titan so the number might slightly change
    EXPECTED_RELATIVE_DIFFERENCE = 2.109659552692574

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John.\nI am a friend of your father.\n")
    MAX_NEW_TOKENS = 10

    def setUp(self):
        """
        Setup quantized model from loaded model
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

        self.bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model_4bit = load_and_quantize_model(self.model_4bit, self.bnb_quantization_config)

        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

    def tearDown(self):
        """
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        super().tearDown()
        del self.model_fp16
        del self.model_4bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Params4bit

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()

        assert round((mem_fp16 / mem_4bit) - self.EXPECTED_RELATIVE_DIFFERENCE, 7) >= 0
        assert self.model_4bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Params4bit

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if (
                    name
                    not in self.bnb_quantization_config.keep_in_fp32_modules
                    + self.bnb_quantization_config.skip_modules
                ):
                    # 4-bit parameters are packed in uint8 variables
                    assert module.weight.dtype == torch.uint8

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        output_sequences = self.model_4bit.generate(
            input_ids=encoded_input["input_ids"].to(self.model_4bit.device), max_new_tokens=10
        )

        assert self.tokenizer.decode(output_sequences[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, keep_in_fp32_modules=["lm_head"])

        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        model = load_and_quantize_model(model, bnb_quantization_config)
        assert model.lm_head.weight.dtype == torch.float32
