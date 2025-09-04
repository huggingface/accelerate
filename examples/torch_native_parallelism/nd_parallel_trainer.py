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
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from accelerate.utils import ParallelismConfig
from utils import get_dataset


MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--checkpoint-frequency", type=int, default=100)
    parser.add_argument("--model-name", type=str, default=MODEL_ID)
    parser.add_argument("--save-dir", type=str, default=f"./accelerate-nd-parallel-{MODEL_ID.split('/')[-1]}")
    parser.add_argument("--device-type", type=str, default="auto")
    return parser.parse_args()


def main():
    # If ParallelismConfig is not initialized with __init__, it reads from env vars
    # which were set by using config
    pc = ParallelismConfig()
    args = parse_args()

    if args.device_type == "auto":
        args.device_type = torch.accelerator.current_accelerator().type

    model_kwargs = {}
    if pc.tp_enabled:
        model_kwargs["tp_plan"] = "auto"
        model_kwargs["device_mesh"] = pc.build_device_mesh(args.device_type)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_dataset = get_dataset(tokenizer, args.sequence_length)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        parallelism_config=pc,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=5,
        save_steps=args.checkpoint_frequency,
        learning_rate=5e-5,
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
