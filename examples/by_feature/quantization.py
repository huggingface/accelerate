# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator


"""
Simple example of 4bit or 8bit quantization with BitsAndBytes
Pure 4bit or 8bit training is not supported. We train adapters on top of the model with peft library.

Steps:
`accelerate config` and choose your quantization parameters
`accelerate launch quantization.py`
"""


def main():
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    with accelerator.main_process_first():
        data = load_dataset("Abirate/english_quotes")
        data = data.map(
            lambda samples: tokenizer(samples["quote"]), batched=True, remove_columns=["quote", "author", "tags"]
        )

    train_dataloader = DataLoader(
        data["train"],
        collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        batch_size=8,
        shuffle=True,
    )

    learning_rate = 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training peft adapters with 8bit or 4bit quantization
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    for epoch in range(10):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                print("Loss:", loss.item())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()


if __name__ == "__main__":
    main()
