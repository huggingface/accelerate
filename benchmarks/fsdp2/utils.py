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

import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer_post_shard",
        action="store_true",
        default=False,
        help="If True, the optimizer will be sharded after applying `fully_shard`",
    )
    parser.add_argument(
        "--optimizer_apply_fix",
        action="store_true",
        default=False,
        help="Only used if `--optimizer_post_shard` is False. If True, the optimizer will be fixed to lower the memory footprint. This fix is used in `Accelerate` to enable bringing your own optimizer.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run, will be used to name the output files inside the directory specified by `--output_dir`",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the benchmarking results.",
    )
    parser.add_argument(
        "--save_memory_snapshot",
        action="store_true",
        default=False,
        help="If True, `torch.cuda.memory._dump_snapshot` will be used to additionaly save the memory trace.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If True, the benchmarking results will be logged to Weights & Biases under the run specified by `--run_name`.",
    )
    ######################
    # Training arguments #
    ######################
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the training loop.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="The maximum sequence length to use with the model.",
    )
    return parser.parse_args()


def prepare_dataloader(tokenizer, args) -> DataLoader:
    dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
        )

    with Accelerator().main_process_first():
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

    block_size = min(tokenizer.model_max_length, args.block_size)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)

    def collate_fn(examples):
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )(examples)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    dataloader = Accelerator().prepare(dataloader)
    return dataloader
