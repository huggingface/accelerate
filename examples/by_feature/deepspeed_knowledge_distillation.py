# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from accelerate import Accelerator, DistributedType


########################################################################
# This is a fully working simple example to use Accelerate to perform knowledge distillation
# across multiple models using DeepSpeed.
#
# Based on: https://www.philschmid.de/knowledge-distillation-bert-transformers
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

STUDENT_ID = "google/bert_uncased_L-2_H-128_A-2"
TEACHER_ID = "textattack/bert-base-uncased-SST-2"


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    # We get the tokenizer from the teacher model
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    datasets = load_dataset("glue", "sst2")

    def tokenize_function(examples):
        # max_length=512 to align lengths
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=512)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # For Torchxla, it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.XLA else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


# def train_knowledge_distillation(teacher, student, optimizer, scheduler, train_loader, epochs):
#     teacher.eval()  # Teacher set to evaluation mode
#     student.train()  # Student to train mode

#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:

#             optimizer.zero_grad()

#             # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
#             with torch.no_grad():
#                 teacher_logits = teacher(inputs)

#             # Forward pass with the student model
#             student_logits = student(inputs)

#             # Soften the student logits by applying softmax first and log() second
#             soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
#             soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

#             # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
#             soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

#             # Calculate the true label loss
#             label_loss = ce_loss(student_logits, labels)

#             # Weighted sum of the two losses
#             loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
# train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
# test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

# # Compare the student test accuracy with and without the teacher, after distillation
# print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
# print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
# print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
