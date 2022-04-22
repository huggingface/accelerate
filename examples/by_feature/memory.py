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
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType

# New Code #
from accelerate.memory_utils import memory_aware
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


########################################################################
# This is a fully working simple example to use Accelerate,
# specifically showcasing how to ensure out-of-memory errors never
# iterrupt training, and builds off the `nlp_example.py` script.
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# New additions from the base script can be found quickly by
# looking for the # New Code # tags
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################

# New Code #
set_seed(42)

# New Code #
# In our particular use case, we rely on the `Accelerator` object being present
# when we want to build the DataLoaders. For our wrapper to work we *must* build
# our `Accelerator` object before we execute the function
accelerator = Accelerator()

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def get_dataloaders(batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader


# New Code #
# If the batch size is too big we use gradient accumulation
gradient_accumulation_steps = 1
batch_size = 16
if batch_size > MAX_GPU_BATCH_SIZE:
    gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
    batch_size = MAX_GPU_BATCH_SIZE

# Our `DataLoaders` need to be built before declaring our main training function in the script
train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)

# Similarly we can also get our model, optimizer, and scheduler to keep our training loop clean
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

# We could avoid this line since the accelerator is set with `device_placement=True` (default value).
# Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
# creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
model = model.to(accelerator.device)

# Instantiate optimizer
optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=(len(train_dataloader) * 3),
)

# Prepare everything
# There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
# prepare method.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

metric = load_metric("glue", "mrpc")

# New Code #
# Now we can define our training loop, wrapping it in `memory_aware`
# This will repeatedly call `training_function` if a CUDA OOM error was hit,
# and reduces all of the batch_sizes in the dataloaders by half


@memory_aware(dataloaders=[train_dataloader, eval_dataloader])
def training_function():
    # Train the model as usual
    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)


def main():
    training_function()


if __name__ == "__main__":
    main()
