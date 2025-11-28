# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from deepspeed.runtime.utils import move_to_device
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import ParallelismConfig, set_seed
from accelerate.utils.dataclasses import DeepSpeedSequenceParallelConfig


set_seed(42)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# to run the example faster switch to the random model
# model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

micro_batch_size = 1

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length=256,
        sp_seq_length_is_variable=True,
        sp_attn_implementation="sdpa",
    ),
)

accelerator = Accelerator(
    parallelism_config=parallelism_config,
    #    log_with="wandb",  # enable to log into wandb
)
accelerator.init_trackers(
    project_name="ulysses-accelerate",
    config={},
    init_kwargs={"wandb": dict(entity="yak", name="deepspeed")},
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2 quick rough datasets to demonstrate the workings
if 1:  # real dataset
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:12]")

    # this is a quick example, it should be made more efficient to be used in real application
    def convert(ex):
        texts = tokenizer.apply_chat_template(conversation=ex["messages"], tokenize=False)
        tokenized_dict = tokenizer(texts, max_length=256, padding=True, truncation=True)
        return tokenized_dict

    ds = ds.map(convert, batched=False, remove_columns=["prompt", "prompt_id", "messages"])

    def collate_fn(batch):
        input_ids = torch.tensor(batch[0]["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch[0]["attention_mask"]).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            attention_mask=attention_mask,
        )

    dl = torch.utils.data.DataLoader(
        ds, batch_size=micro_batch_size, collate_fn=collate_fn, drop_last=True, shuffle=False
    )

else:  # fake dataset
    samples = 16
    seqlen = 256
    input_ids = torch.arange(1, seqlen * samples + 1).view(-1, seqlen) + 100
    position_ids = torch.arange(seqlen * samples).view(-1, seqlen)

    ds = torch.utils.data.TensorDataset(input_ids, position_ids)

    def collate_fn(batch):
        input_ids, position_ids = batch[0]
        return dict(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            labels=input_ids.unsqueeze(0),
        )

    dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

rank = torch.distributed.get_rank()

if rank == 0:
    print(f"DL orig: {len(dl)} samples")

model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

if rank == 0:
    print(f"DL w/ adapter: {len(dl)} samples")

sp_size = parallelism_config.sp_size if parallelism_config else 1
if sp_size > 1:
    sp_group = accelerator.torch_device_mesh["sp"].get_group()
    sp_world_size = parallelism_config.sp_size

unwrapped_model = accelerator.unwrap_model(model)

# Normal training loop
for iter, batch in enumerate(dl):
    optimizer.zero_grad()

    if rank == 0:
        print(f"batch {iter}: seqlen: {len(batch['input_ids'][0])}")
    batch = move_to_device(batch, model.device)

    # The model automatically receives shift_labels via **kwargs and uses it for loss computation.
    # Both standard transformer models and Liger-patched models handle this correctly.
    outputs = model(**batch)
    loss = outputs.loss

    if sp_size > 1:
        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
        # special dealing with SFT that has prompt tokens that aren't used in loss computation
        good_tokens = (batch["shift_labels"] != -100).view(-1).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / max(total_good_tokens, 1)

    if rank == 0:
        accelerator.print(f"{iter}: {loss=}")
    accelerator.log(dict(train_loss=loss, step=iter))

    accelerator.backward(loss)
    optimizer.step()

accelerator.end_training()
