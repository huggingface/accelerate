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
import unittest
import tempfile

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from accelerate.test_utils import require_bnb, require_cuda, require_huggingface_suite, slow
from accelerate.utils.bnb import get_quantized_model
from accelerate.utils.dataclasses import BnbQuantizationConfig


class BitsAndBytesConfigIntegration(unittest.TestCase):
    def test_BnbQuantizationConfig(self):
        with self.assertRaises(ValueError):
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_8bit=True,
                load_in_4bit=True,
            )

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
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

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
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        # create model on meta device
        with init_empty_weights():
            self.model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        self.weights_location = hf_hub_download(self.model_name, "pytorch_model.bin")
        self.bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer
        self.model_8bit = get_quantized_model(
            self.model_8bit, self.bnb_quantization_config, weights_location=self.weights_location, device_map="auto",no_split_module_classes=['BloomBlock']
        )

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

        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

    def test_linear_are_8bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_not_converted = self.bnb_quantization_config.keep_in_fp32_modules + self.bnb_quantization_config.skip_modules
                if name not in modules_not_converted:
                    self.assertTrue(module.weight.dtype == torch.int8)
                    
    def test_llm_skip(self):
        r"""
        A simple test to check if `llm_int8_skip_modules` works as expected
        """
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM,AutoConfig

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, skip_modules=["lm_head","transformer.word_embeddings"])
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        
        model = get_quantized_model(
            model, bnb_quantization_config, weights_location=self.weights_location, device_map="auto", no_split_module_classes=['BloomBlock']
        )
        
        self.assertTrue(model.transformer.h[1].mlp.dense_4h_to_h.weight.dtype == torch.int8)
        self.assertTrue(
            isinstance(model.transformer.h[1].mlp.dense_4h_to_h, bnb.nn.Linear8bitLt)
        )
        self.assertTrue(isinstance(model.lm_head, nn.Linear))
        self.assertTrue(model.lm_head.weight.dtype != torch.int8)
    
    def check_inference_correctness(self,model):
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
        self.assertEqual(output_text, self.EXPECTED_OUTPUT)
        
    def test_generate_quality(self):
        self.check_inference_correctness(self.model_8bit)
        
    def test_fp32_8bit_conversion(self):
        r"""
        Test whether it is possible to mix both `8bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM,AutoConfig
         
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, keep_in_fp32_modules=["lm_head"])
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        
        model = get_quantized_model(
            model, bnb_quantization_config, weights_location=self.weights_location, device_map="auto", no_split_module_classes=['BloomBlock']
        )
        self.assertTrue(model.lm_head.weight.dtype == torch.float32)

    def test_cpu_gpu_loading_random_device_map(self):
        from transformers import AutoModelForCausalLM, AutoConfig
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

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, enable_fp32_cpu_offload=True)

        with init_empty_weights():
            model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer
        model_8bit = get_quantized_model(
            model_8bit, bnb_quantization_config, weights_location=self.weights_location, device_map=device_map,no_split_module_classes=['BloomBlock'])
        self.check_inference_correctness(model_8bit)
        
    def test_cpu_gpu_loading_custom_device_map(self):
        from transformers import AutoModelForCausalLM, AutoConfig
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

        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, enable_fp32_cpu_offload=True)

        with init_empty_weights():
            model_8bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer
        model_8bit = get_quantized_model(
            model_8bit, bnb_quantization_config, weights_location=self.weights_location, device_map=device_map,no_split_module_classes=['BloomBlock'])
        self.check_inference_correctness(model_8bit)
        
        
        
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
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        """
        Setup quantized model from loaded model
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        self.bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True)
        
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model_8bit = get_quantized_model(self.model_8bit,self.bnb_quantization_config)

        self.tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b7')

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

        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_8bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Int8Params)

    def test_linear_are_8bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """

        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_not_converted = self.bnb_quantization_config.keep_in_fp32_modules + self.bnb_quantization_config.skip_modules
                if name not in modules_not_converted:
                    self.assertTrue(module.weight.dtype == torch.int8)
                    
    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        
        output_sequences = self.model_8bit.generate(input_ids=encoded_input["input_ids"].to(self.model_8bit.device), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)
        
    def test_fp32_8bit_conversion(self):
        r"""
        Test whether it is possible to mix both `8bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM
         
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, keep_in_fp32_modules=["lm_head"])
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        model = get_quantized_model(model, bnb_quantization_config)
        self.assertTrue(model.lm_head.weight.dtype == torch.float32)

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
    EXPECTED_RELATIVE_DIFFERENCE = (
        2.109659552692574  # This was obtained on a RTX Titan so the number might slightly change
    )

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
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        # create model on meta device
        with init_empty_weights():
            self.model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        self.weights_location = hf_hub_download(self.model_name, "pytorch_model.bin")
        self.bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer.
        self.model_4bit = get_quantized_model(
            self.model_4bit, self.bnb_quantization_config, weights_location=self.weights_location, device_map="auto",no_split_module_classes=['BloomBlock']
        )

        self.tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b7')

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

        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_4bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Params4bit)

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = self.model_4bit.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)
        
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    
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
                    self.assertTrue(module.weight.dtype == torch.uint8)
                    
    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM,AutoConfig
         
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, keep_in_fp32_modules=["lm_head"])
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        model = get_quantized_model(
            model, bnb_quantization_config, weights_location=self.weights_location, device_map="auto", no_split_module_classes=['BloomBlock']
        )
        self.assertTrue(model.lm_head.weight.dtype == torch.float32)
        
    def test_cpu_gpu_loading_random_device_map(self):
        from transformers import AutoModelForCausalLM, AutoConfig
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

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, enable_fp32_cpu_offload=True)

        with init_empty_weights():
            model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer
        model_4bit = get_quantized_model(
            model_4bit, bnb_quantization_config, weights_location=self.weights_location, device_map=device_map,no_split_module_classes=['BloomBlock'])
        self.check_inference_correctness(model_4bit)
        
    def test_cpu_gpu_loading_custom_device_map(self):
        from transformers import AutoModelForCausalLM, AutoConfig
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

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, enable_fp32_cpu_offload=True)

        with init_empty_weights():
            model_4bit = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))

        # Need to select the transformer layer because the weights at weights_location describe the transformer layer
        model_4bit = get_quantized_model(
            model_4bit, bnb_quantization_config, weights_location=self.weights_location, device_map=device_map,no_split_module_classes=['BloomBlock'])
        self.check_inference_correctness(model_4bit)
        
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
    EXPECTED_RELATIVE_DIFFERENCE = (
        2.109659552692574  # This was obtained on a RTX Titan so the number might slightly change
    )

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
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        self.bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True)

        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model_4bit = get_quantized_model(self.model_4bit, self.bnb_quantization_config)

        self.tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-1b7')

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

        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        self.assertTrue(self.model_4bit.transformer.h[0].mlp.dense_4h_to_h.weight.__class__ == Params4bit)

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
                    self.assertTrue(module.weight.dtype == torch.uint8)
                    
    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        
        output_sequences = self.model_4bit.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)
        
    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        from transformers import AutoModelForCausalLM
         
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, keep_in_fp32_modules=["lm_head"])
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        model = get_quantized_model(model, bnb_quantization_config)
        self.assertTrue(model.lm_head.weight.dtype == torch.float32)