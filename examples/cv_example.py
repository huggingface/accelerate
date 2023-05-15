# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import os
import re

import numpy as np
import PIL
import torch
from timm import create_model
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from accelerate import Accelerator


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a ResNet50 on the Oxford-IIT Pet Dataset
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


# Function to get the label from the filename
def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]


class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    image_size = config["image_size"]
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)

    # Grab all the image filenames
    file_names = [os.path.join(args.data_dir, fname) for fname in os.listdir(args.data_dir) if fname.endswith(".jpg")]

    # Build the label correspondences
    all_labels = [extract_label(fname) for fname in file_names]
    id_to_label = list(set(all_labels))
    id_to_label.sort()
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Split our filenames between train and validation
    random_perm = np.random.permutation(len(file_names))
    cut = int(0.8 * len(file_names))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training we use a simple RandomResizedCrop
    train_tfm = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset(
        [file_names[i] for i in train_split], image_transform=train_tfm, label_to_id=label_to_id
    )

    # For evaluation, we use a deterministic Resize
    eval_tfm = Compose([Resize(image_size), ToTensor()])
    eval_dataset = PetsDataset([file_names[i] for i in eval_split], image_transform=eval_tfm, label_to_id=label_to_id)

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # We normalize the batches of images to be a bit faster.
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None].to(accelerator.device)
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None].to(accelerator.device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr / 25)

    # Instantiate learning rate scheduler
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader))

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        accurate = 0
        num_elems = 0
        for _, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["label"]))
            accurate_preds = predictions == references
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 3e-2, "num_epochs": 3, "seed": 42, "batch_size": 64, "image_size": 224}
    training_function(config, args)


if __name__ == "__main__":
    main()
