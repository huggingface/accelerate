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

import torch

from accelerate.accelerator import Accelerator
from accelerate.test_utils import require_bnb, require_cuda, require_huggingface_suite, require_peft, slow
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils.bitsandbytes import get_bnb_model
from accelerate.utils.dataclasses import BnbQuantizationPlugin


class BitsAndBytesConfigIntegration(unittest.TestCase):
    def test_BitsAndBytes_plugin(self):
        bnb_quantization_plugin = BnbQuantizationPlugin(
            load_in_8bit=True, llm_int8_threshold=6, llm_int8_skip_modules="lm_head"
        )
        self.assertFalse(bnb_quantization_plugin.load_in_4bit)
        with self.assertRaises(ValueError):
            bnb_quantization_plugin = BnbQuantizationPlugin(
                load_in_8bit=True,
                load_in_4bit=True,
            )


@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class BaseMixedInt8Test(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b7"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        1.540025  # This was obtained on a Quadro RTX 8000 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUT = "Hello my name is John.\nI am a friend of the family.\n"
    MAX_NEW_TOKENS = 10

    def setUp(self):
        from transformers import AutoTokenizer

        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


class MixedInt8Test(BaseMixedInt8Test):
    def setUp(self):
        from transformers import AutoModelForCausalLM

        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        # Loading the model using get_bnb model instead of the `prepare` method from `Accelerator`
        # because it will also prepare the model for training.
        bnb_quantization_plugin = BnbQuantizationPlugin(load_in_8bit=True)
        self.model_8bit = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model_8bit = get_bnb_model(self.model_8bit, bnb_quantization_plugin.to_dict(), False).cuda(0)

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_8bit.config

        self.assertTrue(hasattr(config, "quantization_config"))

        _ = config.to_dict()
        _ = config.to_diff_dict()

        _ = config.to_json_string()

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
        from transformers import T5PreTrainedModel

        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ["lm_head"] + T5PreTrainedModel._keep_in_fp32_modules:
                    self.assertTrue(module.weight.dtype == torch.int8)

    def test_generate_quality(self):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")
        output_sequences = self.model_8bit.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        self.assertEqual(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUT)


@slow
@require_cuda
@require_peft
@require_huggingface_suite
class Bnb8BitConversion(AccelerateTestCase):
    def setUp(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = "facebook/opt-350m"
        bnb_quantization_plugin = BnbQuantizationPlugin(load_in_8bit=True)

        self.accelerate = Accelerator(bnb_quantization_plugin=bnb_quantization_plugin)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_8bit_transformers = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True)

    def tearDown(self):
        """
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        super().tearDown()

    def test_compare_quantization(self):
        """
        Test if we have the same model with we do quantization using transformers and accelerate
        """
        from peft import prepare_model_for_kbit_training

        model_8bit_accelerate = self.accelerate.prepare(self.model)
        model_8bit_transformers = prepare_model_for_kbit_training(self.model_8bit_transformers)

        mem_1 = model_8bit_accelerate.get_memory_footprint()
        mem_2 = model_8bit_transformers.get_memory_footprint()
        self.assertTrue(mem_1 == mem_2)

        same = True
        for key_item_1, key_item_2 in zip(
            model_8bit_accelerate.state_dict().items(), model_8bit_transformers.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                same = False
        self.assertTrue(same)

    def test_quantization_peft_model(self):
        """
        Test if we have the same model if we do:
        1: quantization 8bit -> peft
        2: peft -> quantization 8bit
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.model, peft_config)
        bnb_8bit_peft_model_accelerate = self.accelerate.prepare(peft_model)

        bnb_8bit_peft_model_transformers = get_peft_model(self.model_8bit_transformers, peft_config)
        bnb_8bit_peft_model_transformers = prepare_model_for_kbit_training(bnb_8bit_peft_model_transformers)

        mem_1 = bnb_8bit_peft_model_accelerate.get_memory_footprint()
        mem_2 = bnb_8bit_peft_model_transformers.get_memory_footprint()
        self.assertTrue(mem_1 == mem_2)

        same = True
        for key_item_1, key_item_2 in zip(
            bnb_8bit_peft_model_accelerate.state_dict().items(), bnb_8bit_peft_model_transformers.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                # if it is lora layer, the weight are initialized randomly
                if "lora" in key_item_1[0] and key_item_1[0] == key_item_2[0]:
                    pass
                else:
                    same = False
        self.assertTrue(same)


@slow
@require_cuda
@require_bnb
@require_huggingface_suite
class Base4bitTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "bigscience/bloom-1b7"

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
        from transformers import AutoTokenizer

        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


class Bnb4BitTest(Base4bitTest):
    def setUp(self):
        from transformers import AutoModelForCausalLM

        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)

        # Loading the model using get_bnb model instead of the `prepare` method from `Accelerator`
        # because it will also prepare the model for training.
        bnb_quantization_plugin = BnbQuantizationPlugin(load_in_4bit=True)
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model_4bit = get_bnb_model(self.model_4bit, bnb_quantization_plugin.to_dict(), False).cuda(0)

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

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_4bit.config

        self.assertTrue(hasattr(config, "quantization_config"))

        _ = config.to_dict()
        _ = config.to_diff_dict()

        _ = config.to_json_string()

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
        from transformers import T5PreTrainedModel

        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ["lm_head"] + T5PreTrainedModel._keep_in_fp32_modules:
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


@slow
@require_cuda
@require_peft
@require_huggingface_suite
class Bnb4BitConversion(AccelerateTestCase):
    def setUp(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = "facebook/opt-350m"
        bnb_quantization_plugin = BnbQuantizationPlugin(load_in_4bit=True)

        self.accelerate = Accelerator(bnb_quantization_plugin=bnb_quantization_plugin)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_4bit_transformers = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)

    def tearDown(self):
        """
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        super().tearDown()

    def test_compare_quantization(self):
        """
        Test if we have the same model with we do quantization using transformers and accelerate
        """
        from peft import prepare_model_for_kbit_training

        model_4bit_accelerate = self.accelerate.prepare(self.model)
        model_4bit_transformers = prepare_model_for_kbit_training(self.model_4bit_transformers)

        mem_1 = model_4bit_accelerate.get_memory_footprint()
        mem_2 = model_4bit_transformers.get_memory_footprint()
        self.assertTrue(mem_1 == mem_2)

        same = True
        for key_item_1, key_item_2 in zip(
            model_4bit_accelerate.state_dict().items(), model_4bit_transformers.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                same = False
        self.assertTrue(same)

    def test_quantize_peft_model(self):
        """
        Test if we have the same model if we do:
        1: quantization 4bit -> peft
        2: peft -> quantization 4bit
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.model, peft_config)
        bnb_4bit_peft_model_accelerate = self.accelerate.prepare(peft_model)

        bnb_4bit_peft_model_transformers = get_peft_model(self.model_4bit_transformers, peft_config)
        bnb_4bit_peft_model_transformers = prepare_model_for_kbit_training(bnb_4bit_peft_model_transformers)

        mem_1 = bnb_4bit_peft_model_accelerate.get_memory_footprint()
        mem_2 = bnb_4bit_peft_model_transformers.get_memory_footprint()
        self.assertTrue(mem_1 == mem_2)

        same = True
        for key_item_1, key_item_2 in zip(
            bnb_4bit_peft_model_accelerate.state_dict().items(), bnb_4bit_peft_model_transformers.state_dict().items()
        ):
            if not torch.equal(key_item_1[1], key_item_2[1]):
                # if it is lora layer, the weights are initialized randomly
                if "lora" in key_item_1[0] and key_item_1[0] == key_item_2[0]:
                    pass
                else:
                    same = False
        self.assertTrue(same)
