# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from datetime import timedelta

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin


class LmHeadWrapper(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, x):
        return self.lm_head(x)


def build_simple_dataloader(tokenizer, seq_len=64, batch_size=2):
    """Build a simple dataloader for reproduction."""
    # Load small dataset
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    raw = raw.filter(lambda x: len(tokenizer(x["text"])["input_ids"]) > 0)
    raw = raw.select(range(min(100, len(raw))))  # Use only 100 samples

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len)

    ds = raw.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids"])

    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        labels = [x.clone() for x in ids]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        x = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        y = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": x, "labels": y}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    BATCH_SIZE = 2
    SEQ_LEN = 64
    TP = 2
    DP = 4 // TP

    # Setup Accelerator with FSDP2
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    pc = ParallelismConfig(dp_shard_size=DP, tp_size=TP)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        auto_wrap_policy="transformer_based_wrap",
        state_dict_type="SHARDED_STATE_DICT",
        activation_checkpointing=False,
        cpu_ram_efficient_loading=True,
    )

    accelerator = Accelerator(kwargs_handlers=[init_kwargs], parallelism_config=pc, fsdp_plugin=fsdp_plugin)

    rank = accelerator.process_index
    print(f"[Rank {rank}] Initializing...")

    # Load model with TP if needed
    model_kwargs = {"tp_size": TP, "tp_plan": "auto", "device_mesh": accelerator.torch_device_mesh} if TP > 1 else {}

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_cache=False, **model_kwargs)

    model.lm_head = LmHeadWrapper(model.lm_head)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"[Rank {rank}] Building dataloader...")
    loader = build_simple_dataloader(tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    print(f"[Rank {rank}] Preparing with accelerator...")
    # ERROR OCCURS HERE AT LINE 110 in original script
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    print(f"[Rank {rank}] Preparation successful!")


if __name__ == "__main__":
    main()
