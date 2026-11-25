# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import argparse

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin


MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_ram_efficient_loading", choices=["false", "true"], default="false")
    parser.add_argument("--cpu_offload", choices=["false", "true"], default="false")
    parser.add_argument("--ignored_modules", choices=["false", "true"], default="false")
    args = parser.parse_args()

    # Load before distributed initialization so every rank has a complete reference.
    reference = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        cpu_ram_efficient_loading=args.cpu_ram_efficient_loading == "true",
        cpu_offload=args.cpu_offload == "true",
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    # Qwen already has non-persistent rotary buffers; add a persistent buffer too.
    for module in (model, reference):
        module.register_buffer("persistent_buffer", torch.arange(16, dtype=torch.float32).reshape(4, 4).T)
    reference.to(accelerator.device)

    if args.ignored_modules == "true":
        fsdp_plugin.ignored_modules = "lm_head"
    model = accelerator.prepare(model)

    assert isinstance(model.get_input_embeddings().weight, DTensor)
    if args.ignored_modules == "true":
        assert not isinstance(model.lm_head.weight, DTensor)
    else:
        assert isinstance(model.lm_head.weight, DTensor)
    expected = reference.state_dict()
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, DTensor):
            tensor = tensor.to(accelerator.device).full_tensor()
        torch.testing.assert_close(tensor, expected[name])
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, reference.get_buffer(name))

    inputs = torch.ones(1, 4, dtype=torch.long, device=accelerator.device)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).logits, reference(inputs).logits)

    accelerator.end_training()


if __name__ == "__main__":
    main()
