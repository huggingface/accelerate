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
from measure_utils import MemoryTracker
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from accelerate import Accelerator
from utils import parse_args, prepare_dataloader


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LEARNING_RATE = 1e-4


def replace_optimizer_params(optimizer):
    for param_group in optimizer.param_groups:
        for i, p in enumerate(param_group["params"]):
            param_group["params"][i] = torch.empty_like(p)
            param_group["params"][i].data_ptr = p.data_ptr()


def swap_back_optimizer_params(
    accelerator: Accelerator, model: torch.nn.Module, optimizer: torch.optim.Optimizer, old_named_parameters: dict
):
    new_named_parameters = accelerator._get_named_parameters(model)

    mapping = {p: new_named_parameters[n] for n, p in old_named_parameters.items()}

    for param_group in optimizer.param_groups:
        param_group["params"] = [mapping.get(p, p) for p in param_group["params"]]


def main():
    args = parse_args()
    accelerator = Accelerator()

    optimizer = None

    if args.wandb:
        accelerator.init_trackers(
            "FSDP2-Benchmark",
            init_kwargs={"wandb": {"name": args.wandb_run}},
        )

    memory_tracker = MemoryTracker(
        device=accelerator.device,
        output_directory=args.output_dir,
        run_name=args.run_name,
        save_memory_snapshot=args.save_memory_snapshot,
    )
    memory_tracker.start()

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader = prepare_dataloader(tokenizer, args)

    if not args.optimizer_post_shard:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    if args.optimizer_apply_fix:
        old_named_parameters = accelerator._get_named_parameters(model)
        replace_optimizer_params(optimizer)

    for module in model.modules():
        if isinstance(module, Qwen2DecoderLayer):
            fully_shard(module)
    fully_shard(model)

    if args.optimizer_post_shard:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    if args.optimizer_apply_fix:
        swap_back_optimizer_params(accelerator, model, optimizer, old_named_parameters)

    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(**batch, use_cache=False)

        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

        if args.wandb:
            accelerator.log({"loss": loss.item()}, step=step)

    memory_tracker.stop()
    accelerator.end_training()


if __name__ == "__main__":
    main()
