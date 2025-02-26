# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch.distributed

from accelerate.test_utils import require_huggingface_suite, torch_device
from accelerate.utils import is_transformers_available


if is_transformers_available():
    from transformers import AutoModel, TrainingArguments


GPT2_TINY = "sshleifer/tiny-gpt2"


@require_huggingface_suite
def init_torch_dist_then_launch_deepspeed():
    if torch_device == "xpu":
        backend = "ccl"
    elif torch_device == "hpu":
        backend = "hccl"
    else:
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend)
    deepspeed_config = {
        "zero_optimization": {
            "stage": 3,
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }
    train_args = TrainingArguments(
        output_dir="./",
        deepspeed=deepspeed_config,
    )
    model = AutoModel.from_pretrained(GPT2_TINY)
    assert train_args is not None
    assert model is not None


def main():
    init_torch_dist_then_launch_deepspeed()


if __name__ == "__main__":
    main()
