# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import math

import torch


def pack_tokens(token_ids, sequence_length):
    """Drop the final incomplete sequence so every case processes identical dense shapes."""
    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    usable = tokens.numel() // sequence_length * sequence_length
    if not usable:
        raise ValueError("The corpus must contain at least one complete sequence.")
    return tokens[:usable].reshape(-1, sequence_length)


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return {"input_ids": tokens, "labels": tokens}


def get_training_utilities(args, accelerator):
    from datasets import load_dataset
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def packed_split(split):
        text = tokenizer.eos_token.join(line for line in corpus[split]["text"] if line.strip())
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
        return pack_tokens(ids, args.sequence_length)

    train_tokens = packed_split("train")
    eval_tokens = packed_split("validation")[: args.eval_sequences]
    if len(train_tokens) < args.batch_size:
        raise ValueError("The corpus is too small for a complete training batch.")
    train_loader = DataLoader(
        PackedDataset(train_tokens),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    # Fixed evaluation batch size makes the quality comparison independent of the training batch sweep.
    eval_loader = DataLoader(PackedDataset(eval_tokens), batch_size=1)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    # Fused AdamW avoids foreach's extra parameter-sized intermediates; all cases retain FP32 moments.
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.warmup_steps + args.steps,
    )
    train_loader, eval_loader = accelerator.prepare(train_loader, eval_loader)
    return model, optimizer, train_loader, eval_loader, scheduler


def evaluate_model(model, dataloader):
    """Evaluate held-out next-token quality in BF16, with FP8 disabled by eval mode."""
    was_training = model.training
    model.eval()
    total_loss = correct = count = 0
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in dataloader:
                outputs = model(**batch)
                targets = batch["labels"][:, 1:]
                tokens = targets.numel()
                total_loss += outputs.loss.float().item() * tokens
                correct += (outputs.logits[:, :-1].argmax(dim=-1) == targets).sum().item()
                count += tokens
        loss = total_loss / count
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite held-out language-model loss.")
        return {"loss": loss, "perplexity": math.exp(loss), "token_accuracy": correct / count}
    finally:
        model.train(was_training)
