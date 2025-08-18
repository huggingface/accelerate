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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from accelerate.utils import set_seed
from utils import get_dataset


def main():
    set_seed(42)
    model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
    seq_len = 1024

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_dataset = get_dataset(tokenizer, seq_len)

    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=5,
        save_steps=1000,
        learning_rate=5e-5,
        remove_unused_columns=False,
        seed=42,
        bf16=True,
        data_seed=42,
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
