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

"""
Common utilities for FSDP2 examples.
"""

import time

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator


def get_dataset(
    accelerator: Accelerator,
    tokenizer: AutoTokenizer,
    seq_len: int,
    processing_batch_size: int = 1000,
) -> Dataset:
    """
    Load and prepare TinyStories dataset.

    Args:
        accelerator (Accelerator): Accelerate accelerator instance
        tokenizer (AutoTokenizer): Hugging Face tokenizer
        seq_len (int): Sequence length for the dataset
        processing_batch_size (int): Batch size for processing the dataset

    Returns:
        Dataset: Packed dataset
    """
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:50%]")

    def tokenize_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=seq_len,
            return_tensors=None,
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"], batch_size=processing_batch_size
        )

    def create_packed_sequences(examples):
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        num_sequences = len(all_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            full_sequence = all_tokens[start_idx:end_idx]
            packed_input_ids.append(full_sequence[:-1])
            packed_labels.append(full_sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    with accelerator.main_process_first():
        packed_dataset = tokenized_dataset.map(
            create_packed_sequences,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            batch_size=processing_batch_size,
        )

    return packed_dataset.shuffle(seed=42)


def get_model_flops_per_token(model: AutoModelForCausalLM, seq_len: int) -> float:
    """
    Get the number of flops per token for the model.

    Args:
        model (AutoModelForCausalLM): Model to get the flops for
        seq_len (int): Sequence length
    """
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # MLP: 3 matmuls
    mlp_flops = 18 * cfg.hidden_size * cfg.intermediate_size

    # Attn (w/o dotproduct)
    attn_flops = 12 * head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads)

    # attn (dotproduct) - this scales quadratically with sequence length
    attn_dotproduct_flops = 12 * cfg.num_attention_heads * head_dim * seq_len

    # we also ignore embeddings and layernorms, etc
    return (mlp_flops + attn_flops + attn_dotproduct_flops) * cfg.num_hidden_layers


def create_collate_fn():
    """Create a collate function for batching."""

    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    return collate_fn


class PerformanceTracker:
    """Track training performance metrics."""

    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """Reset all tracking variables."""
        self.start_time = None
        self.num_tokens = 0
        self.is_in_warmup = True
        self.step_count = 0

    def step(self, batch_tokens: int) -> dict:
        """
        Update performance tracking with a new step.

        Args:
            batch_tokens (int): Number of tokens in current batch

        Returns:
            dict: Performance metrics if past warmup, empty dict otherwise
        """
        self.step_count += 1

        if self.step_count == self.warmup_steps:
            self.start_time = time.perf_counter()
            self.num_tokens = 0
            self.is_in_warmup = False
            return {"warmup_completed": True}

        if not self.is_in_warmup and self.start_time is not None:
            self.num_tokens += batch_tokens
            total_time = time.perf_counter() - self.start_time
            steps_from_warmup = self.step_count - self.warmup_steps

            if total_time > 0 and steps_from_warmup > 0:
                return {
                    "tokens_per_second": self.num_tokens / total_time,
                    "steps_per_second": steps_from_warmup / total_time,
                    "total_tokens": self.num_tokens,
                    "total_time": total_time,
                }

        return {}


def setup_tokenizer(model_id: str) -> AutoTokenizer:
    """Setup tokenizer with proper padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
