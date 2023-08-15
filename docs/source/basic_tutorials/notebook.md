<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Launching Multi-GPU Training from a Jupyter Environment

This tutorial teaches you how to fine tune a computer vision model with 🤗 Accelerate from a Jupyter Notebook on a distributed system.
You will also learn how to setup a few requirements needed for ensuring your environment is configured properly, your data has been prepared properly, and finally how to launch training.

<Tip>

    This tutorial is also available as a Jupyter Notebook [here](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb)

</Tip>

## Configuring the Environment

Before any training can be performed, a 🤗 Accelerate config file must exist in the system. Usually this can be done by running the following in a terminal and answering the prompts:

```bash
accelerate config
```

However, if general defaults are fine and you are *not* running on a TPU, 🤗Accelerate has a utility to quickly write your GPU configuration into a config file via [`utils.write_basic_config`].

The following code will restart Jupyter after writing the configuration, as CUDA code was called to perform this. 

<Tip warning={true}>

    CUDA can't be initialized more than once on a multi-GPU system. It's fine to debug in the notebook and have calls to CUDA, but in order to finally train a full cleanup and restart will need to be performed.
    
</Tip>

```python
import os
from accelerate.utils import write_basic_config

write_basic_config()  # Write a config file
os._exit(00)  # Restart the notebook
```

## Preparing the Dataset and Model

Next you should prepare your dataset. As mentioned at earlier, great care should be taken when preparing the `DataLoaders` and model to make sure that **nothing** is put on *any* GPU. 

If you do, it is recommended to put that specific code into a function and call that from within the notebook launcher interface, which will be shown later. 

Make sure the dataset is downloaded based on the directions [here](https://github.com/huggingface/accelerate/tree/main/examples#simple-vision-example)

```python
import os, re, torch, PIL
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from accelerate import Accelerator
from accelerate.utils import set_seed
from timm import create_model
```

First you need to create a function to extract the class name based on a filename:

```python
import os

data_dir = "../../images"
fnames = os.listdir(data_dir)
fname = fnames[0]
print(fname)
```

```python out
beagle_32.jpg
```

In the case here, the label is `beagle`. Using regex you can extract the label from the filename:

```python
import re


def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]
```

```python
extract_label(fname)
```

And you can see it properly returned the right name for our file:

```python out
"beagle"
```

Next a `Dataset` class should be made to handle grabbing the image and the label:

```python
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
```

Now to build the dataset. Outside the training function you can find and declare all the filenames and labels and use them as references inside the 
launched function:

```python
fnames = [os.path.join("../../images", fname) for fname in fnames if fname.endswith(".jpg")]
```

Next gather all the labels:

```python
all_labels = [extract_label(fname) for fname in fnames]
id_to_label = list(set(all_labels))
id_to_label.sort()
label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}
```

Next, you should make a `get_dataloaders` function that will return your built dataloaders for you. As mentioned earlier, if data is automatically 
sent to the GPU or a TPU device when building your `DataLoaders`, they must be built using this method. 

```python
def get_dataloaders(batch_size: int = 64):
    "Builds a set of dataloaders with a batch_size"
    random_perm = np.random.permutation(len(fnames))
    cut = int(0.8 * len(fnames))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training a simple RandomResizedCrop will be used
    train_tfm = Compose([RandomResizedCrop((224, 224), scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset([fnames[i] for i in train_split], image_transform=train_tfm, label_to_id=label_to_id)

    # For evaluation a deterministic Resize will be used
    eval_tfm = Compose([Resize((224, 224)), ToTensor()])
    eval_dataset = PetsDataset([fnames[i] for i in eval_split], image_transform=eval_tfm, label_to_id=label_to_id)

    # Instantiate dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=4)
    return train_dataloader, eval_dataloader
```

Finally, you should import the scheduler to be used later:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

## Writing the Training Function

Now you can build the training loop. [`notebook_launcher`] works by passing in a function to call that will be ran across the distributed system.

Here is a basic training loop for the animal classification problem:

<Tip>

    The code has been split up to allow for explainations on each section. A full version that can be copy and pasted will be available at the end

</Tip>


```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    accelerator = Accelerator(mixed_precision=mixed_precision)
```

First you should set the seed and create an [`Accelerator`] object as early in the training loop as possible.

<Tip warning={true}>

    If training on the TPU, your training loop should take in the model as a parameter and it should be instantiated 
    outside of the training loop function. See the [TPU best practices](../concept_guides/training_tpu) 
    to learn why

</Tip>

Next you should build your dataloaders and create your model:

```python
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))
```

<Tip>

    You build the model here so that the seed also controls the new weight initialization

</Tip>

As you are performing transfer learning in this example, the encoder of the model starts out frozen so the head of the model can be 
trained only initially:

```python
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
```

Normalizing the batches of images will make training a little faster:

```python
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]
```

To make these constants available on the active device, you should set it to the Accelerator's device:

```python
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)
```

Next instantiate the rest of the PyTorch classes used for training:

```python
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))
```

Before passing everything to [`~Accelerator.prepare`].

<Tip>

    There is no specific order to remember, you just need to unpack the objects in the same order you gave them to the prepare method.

</Tip>

```python
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
```

Now train the model:

```python
    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
```

The evaluation loop will look slightly different compared to the training loop. The number of elements passed as well as the overall 
total accuracy of each batch will be added to two constants:

```python
        model.eval()
        accurate = 0
        num_elems = 0
```

Next you have the rest of your standard PyTorch loop:

```python
        for batch in eval_dataloader:
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
```

Before finally the last major difference. 

When performing distributed evaluation, the predictions and labels need to be passed through 
[`~Accelerator.gather`] so that all of the data is available on the current device and a properly calculated metric can be achieved:

```python
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()
```

Now you just need to calculate the actual metric for this problem, and you can print it on the main process using [`~Accelerator.print`]:

```python
        eval_metric = accurate.item() / num_elems
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```

A full version of this training loop is available below:

```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Build dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)

    # Instantiate the model (you build the model here so that the seed also controls new weight initaliziations)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # You can normalize the batches of images to be a bit faster
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]

    # To make these constants available on the active device, set it to the accelerator device
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    # Intantiate the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)

    # Instantiate the learning rate scheduler
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the objects in the same order you gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Now you train the model
    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
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
        for batch in eval_dataloader:
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```

## Using the notebook_launcher

All that's left is to use the [`notebook_launcher`].

You pass in the function, the arguments (as a tuple), and the number of processes to train on. (See the [documentation](../package_reference/launchers) for more information)

```python
from accelerate import notebook_launcher
```

```python
args = ("fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=2)
```

In the case of running on the TPU, it would look like so:

```python
model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

args = (model, "fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=8)
```

As it's running it will print the progress as well as state how many devices you ran on. This tutorial was ran with two GPUs:

```python out
Launching training on 2 GPUs.
epoch 0: 88.12
epoch 1: 91.73
epoch 2: 92.58
epoch 3: 93.90
epoch 4: 94.71
```

And that's it!

## Debugging 

A common issue when running the `notebook_launcher` is receiving a CUDA has already been initialized issue. This usually stems
from an import or prior code in the notebook that makes a call to the PyTorch `torch.cuda` sublibrary. To help narrow down what went wrong,
you can launch the `notebook_launcher` with `ACCELERATE_DEBUG_MODE=yes` in your environment and an additional check
will be made when spawning that a regular process can be created and utilize CUDA without issue. (Your CUDA code can still be ran afterwards).

## Conclusion

This notebook showed how to perform distributed training from inside of a Jupyter Notebook. Some key notes to remember:

- Make sure to save any code that use CUDA (or CUDA imports) for the function passed to [`notebook_launcher`]
- Set the `num_processes` to be the number of devices used for training (such as number of GPUs, CPUs, TPUs, etc)
- If using the TPU, declare your model outside the training loop function
