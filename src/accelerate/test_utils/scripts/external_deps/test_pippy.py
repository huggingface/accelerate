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
import torch
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
from accelerate.utils import DistributedType, set_seed


model_to_config = {
    "t5": (T5ForConditionalGeneration, T5Config, 1024),
    "bert": (BertForMaskedLM, BertConfig, 512),
    "gpt2": (GPT2ForSequenceClassification, GPT2Config, 1024),
}


def get_model_and_data(model_name, device, num_processes: int = 2):
    initializer, config, seq_len = model_to_config[model_name]
    config = config()
    model = initializer(config)
    return model, torch.randint(
        low=0,
        high=config.vocab_size,
        size=(num_processes, seq_len),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def test_gpt2_gpu_trace():
    set_seed(42)
    state = PartialState()
    model, inputs = get_model_and_data("gpt2", "cuda:0", state.num_processes)
    model.to("cuda:0")
    model = prepare_pippy(model, example_args=(inputs,))
    with torch.no_grad():
        output = model(inputs)
    # Zach: Check that we just grab the real outputs we need at the end
    if not state.is_last_process:
        assert output is None, "Output was not generated on just the last process!"
    else:
        assert output is not None, "Output was not generated in the last process!"


def test_gpt2_cpu_trace():
    set_seed(42)
    state = PartialState()
    model, inputs = get_model_and_data("gpt2", "cpu", state.num_processes)
    model = prepare_pippy(model, example_args=(inputs,))
    # For inference args need to be a tuple
    inputs = inputs.to("cuda")
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
        state.print("Testing GPT2")
        #    test_gpt2_gpu_trace()
        test_gpt2_cpu_trace()
    else:
        print("Less than two GPUs found, not running tests!")
