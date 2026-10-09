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

from __future__ import annotations

import json

import torch
import torch.distributed as dist
from transformers import AutoModel, BertConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin


def main() -> None:
    config = BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    reference = AutoModel.from_config(config)
    reference.eval()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = reference(input_ids=inputs).last_hidden_state

    plugin = DeepSpeedPlugin(
        hf_ds_config={
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
            },
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
        },
        zero3_init_flag=True,
    )
    accelerator = Accelerator(deepspeed_plugin=plugin)
    rank: int = accelerator.process_index
    try:
        if accelerator.num_processes != 2 or accelerator.device.type != "cuda":
            raise RuntimeError("This probe requires two CUDA ranks.")
        torch.manual_seed(0)
        with plugin.zero3_init_context_manager(enable=False):
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "phase": "inside",
                        "plugin": plugin.is_zero3_init_enabled(),
                        "transformers": is_deepspeed_zero3_enabled(),
                    }
                ),
                flush=True,
            )
            model = AutoModel.from_config(config)
            model.eval()
            model.to(device=accelerator.device)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "embedding_shape": list(model.embeddings.word_embeddings.weight.shape),
                        "partitioned": hasattr(model.embeddings.word_embeddings.weight, "ds_id"),
                    }
                ),
                flush=True,
            )
            with torch.no_grad():
                actual = model(input_ids=inputs.to(device=accelerator.device)).last_hidden_state
            torch.testing.assert_close(actual=actual.cpu(), expected=expected, rtol=1e-4, atol=1e-5)
            assert not is_deepspeed_zero3_enabled()
            assert plugin.deepspeed_config["zero_optimization"]["stage"] == 3

        assert plugin.is_zero3_init_enabled()
        assert is_deepspeed_zero3_enabled()
        partitioned = AutoModel.from_config(config)
        assert hasattr(partitioned.embeddings.word_embeddings.weight, "ds_id")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "forward_shape": list(actual.shape),
                    "forward_matches_cpu": True,
                    "restored_zero3": True,
                    "outside_partitioned": True,
                }
            ),
            flush=True,
        )
        accelerator.wait_for_everyone()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
