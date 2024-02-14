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
import torch
from torchvision.models import resnet34
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2ForSequenceClassification,
    T5Config,
    T5ForConditionalGeneration,
)

from accelerate import PartialState
from accelerate.inference import prepare_pippy
from accelerate.utils import DistributedType, send_to_device, set_seed


model_to_config = {
    "t5": (T5ForConditionalGeneration, T5Config, 1024),
    "bert": (BertForMaskedLM, BertConfig, 512),
    "gpt2": (GPT2ForSequenceClassification, GPT2Config, 1024),
}


def get_model_and_data_for_text(model_name, device, num_processes: int = 2):
    initializer, config, seq_len = model_to_config[model_name]
    config_args = {}
    # Eventually needed for batch inference tests on gpt-2 when bs != 1
    # if model_name == "gpt2":
    #     config_args["pad_token_id"] = 0
    model_config = config(**config_args)
    model = initializer(model_config)
    return model, torch.randint(
        low=0,
        high=model_config.vocab_size,
        size=(num_processes, seq_len),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def test_gpt2(batch_size: int = 2):
    set_seed(42)
    state = PartialState()
    model, inputs = get_model_and_data_for_text("gpt2", "cpu", batch_size)
    model = prepare_pippy(model, example_args=(inputs,), no_split_module_classes=model._no_split_modules)
    # For inference args need to be a tuple
    inputs = inputs.to("cuda")
    with torch.no_grad():
        output = model(inputs)
    # Zach: Check that we just grab the real outputs we need at the end
    if not state.is_last_process:
        assert output is None, "Output was not generated on just the last process!"
    else:
        assert output is not None, "Output was not generated in the last process!"


def test_t5(batch_size: int = 2):
    set_seed(42)
    state = PartialState()
    model, inputs = get_model_and_data_for_text("t5", "cpu", batch_size)
    example_inputs = {"input_ids": inputs, "decoder_input_ids": inputs}
    model = prepare_pippy(
        model,
        no_split_module_classes=model._no_split_modules,
        example_kwargs=example_inputs,
    )
    # For inference args need to be a tuple
    inputs = send_to_device(example_inputs, "cuda:0")
    with torch.no_grad():
        output = model(*inputs.values())
    # Zach: Check that we just grab the real outputs we need at the end
    if not state.is_last_process:
        assert output is None, "Output was not generated on just the last process!"
    else:
        assert output is not None, "Output was not generated in the last process!"


def test_resnet(batch_size: int = 2):
    set_seed(42)
    state = PartialState()
    model = resnet34()
    input_tensor = torch.rand(batch_size, 3, 224, 224)
    model = prepare_pippy(
        model,
        example_args=(input_tensor,),
    )
    inputs = send_to_device(input_tensor, "cuda:0")
    with torch.no_grad():
        output = model(inputs)
    # Zach: Check that we just grab the real outputs we need at the end
    if not state.is_last_process:
        assert output is None, "Output was not generated on just the last process!"
    else:
        assert output is not None, "Output was not generated in the last process!"


if __name__ == "__main__":
    state = PartialState()
    state.print("Testing pippy integration...")
    if state.distributed_type == DistributedType.MULTI_GPU:
        state.print("Testing GPT2...")
        test_gpt2()
        # Issue: When modifying the tokenizer for batch GPT2 inference, there's an issue
        # due to references
        # NameError: cannot access free variable 'chunk_args_list' where it is not associated with a value in enclosing scope
        # test_gpt2(3)
        state.print("Testing T5...")
        test_t5()
        test_t5(1)
        test_t5(3)
        state.print("Testing CV model...")
        test_resnet()
        test_resnet(3)
    else:
        print("Less than two GPUs found, not running tests!")
