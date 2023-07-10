<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# In this folder we showcase various full examples using 🤗 Accelerate

## Simple NLP example

The [nlp_example.py](./nlp_example.py) script is a simple example to train a Bert model on a classification task ([GLUE's MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)).

Prior to running it you should install 🤗 Dataset and 🤗 Transformers:

```bash
pip install datasets evaluate transformers
```

The same script can be run in any of the following configurations:
- single CPU or single GPU
- multi GPUs (using PyTorch distributed mode)
- (multi) TPUs
- fp16 (mixed-precision) or fp32 (normal precision)

To run it in each of these various modes, use the following commands:
- single CPU:
    * from a server without GPU
        ```bash
        python ./nlp_example.py
        ```
    * from any server by passing `cpu=True` to the `Accelerator`.
        ```bash
        python ./nlp_example.py --cpu
        ```
    * from any server with Accelerate launcher
        ```bash
        accelerate launch --cpu ./nlp_example.py
        ```
- single GPU:
    ```bash
    python ./nlp_example.py  # from a server with a GPU
    ```
- with fp16 (mixed-precision)
    * from any server by passing `mixed_precison=fp16` to the `Accelerator`.
        ```bash
        python ./nlp_example.py --mixed_precision fp16
        ```
    * from any server with Accelerate launcher
        ```bash
        accelerate launch --mixed_precision fp16 ./nlp_example.py
- multi GPUs (using PyTorch distributed mode)
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on your server
        accelerate launch ./nlp_example.py  # This will run the script on your server
        ```
    * With traditional PyTorch launcher (`torch.distributed.launch` can be used with older versions of PyTorch)
        ```bash
        python -m torchrun --nproc_per_node 2 --use_env ./nlp_example.py
        ```
- multi GPUs, multi node (several machines, using PyTorch distributed mode)
    * With Accelerate config and launcher, on each machine:
        ```bash
        accelerate config  # This will create a config file on each server
        accelerate launch ./nlp_example.py  # This will run the script on each server
        ```
    * With PyTorch launcher only (`torch.distributed.launch` can be used in older versions of PyTorch)
        ```bash
        python -m torchrun --nproc_per_node 2 \
            --use_env \
            --node_rank 0 \
            --master_addr master_node_ip_address \
            ./nlp_example.py  # On the first server
        python -m torchrun --nproc_per_node 2 \
            --use_env \
            --node_rank 1 \
            --master_addr master_node_ip_address \
            ./nlp_example.py  # On the second server
        ```
- (multi) TPUs
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on your TPU server
        accelerate launch ./nlp_example.py  # This will run the script on each server
        ```
    * In PyTorch:
        Add an `xmp.spawn` line in your script as you usually do.


## Simple vision example

The [cv_example.py](./cv_example.py) script is a simple example to fine-tune a ResNet-50 on a classification task ([Ofxord-IIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)).

The same script can be run in any of the following configurations:
- single CPU or single GPU
- multi GPUs (using PyTorch distributed mode)
- (multi) TPUs
- fp16 (mixed-precision) or fp32 (normal precision)

Prior to running it you should install timm and torchvision:

```bash
pip install timm torchvision
```

and you should download the data with the following commands:

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xzf images.tar.gz
```

To run it in each of these various modes, use the following commands:
- single CPU:
    * from a server without GPU
        ```bash
        python ./cv_example.py --data_dir path_to_data
        ```
    * from any server by passing `cpu=True` to the `Accelerator`.
        ```bash
        python ./cv_example.py --data_dir path_to_data --cpu
        ```
    * from any server with Accelerate launcher
        ```bash
        accelerate launch --cpu ./cv_example.py --data_dir path_to_data
        ```
- single GPU:
    ```bash
    python ./cv_example.py  # from a server with a GPU
    ```
- with fp16 (mixed-precision)
    * from any server by passing `mixed_precison=fp16` to the `Accelerator`.
        ```bash
        python ./cv_example.py --data_dir path_to_data --mixed_precison fp16
        ```
    * from any server with Accelerate launcher
        ```bash
        accelerate launch --mixed_precison fp16 ./cv_example.py --data_dir path_to_data
- multi GPUs (using PyTorch distributed mode)
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on your server
        accelerate launch ./cv_example.py --data_dir path_to_data  # This will run the script on your server
        ```
    * With traditional PyTorch launcher (`torch.distributed.launch` can be used with older versions of PyTorch)
        ```bash
        python -m torchrun --nproc_per_node 2 --use_env ./cv_example.py --data_dir path_to_data
        ```
- multi GPUs, multi node (several machines, using PyTorch distributed mode)
    * With Accelerate config and launcher, on each machine:
        ```bash
        accelerate config  # This will create a config file on each server
        accelerate launch ./cv_example.py --data_dir path_to_data  # This will run the script on each server
        ```
    * With PyTorch launcher only (`torch.distributed.launch` can be used with older versions of PyTorch)
        ```bash
        python -m torchrun --nproc_per_node 2 \
            --use_env \
            --node_rank 0 \
            --master_addr master_node_ip_address \
            ./cv_example.py --data_dir path_to_data  # On the first server
        python -m torchrun --nproc_per_node 2 \
            --use_env \
            --node_rank 1 \
            --master_addr master_node_ip_address \
            ./cv_example.py --data_dir path_to_data  # On the second server
        ```
- (multi) TPUs
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on your TPU server
        accelerate launch ./cv_example.py --data_dir path_to_data  # This will run the script on each server
        ```
    * In PyTorch:
        Add an `xmp.spawn` line in your script as you usually do.

### Simple vision example (GANs)

- [huggan project](https://github.com/huggingface/community-events/tree/main/huggan)

### Using AWS SageMaker integration
- [Examples showcasing AWS SageMaker integration of 🤗 Accelerate.](https://github.com/pacman100/accelerate-aws-sagemaker)


## Simple Multi-GPU Hardware Launcher

[multigpu_remote_launcher.py](./multigpu_remote_launcher.py) is a minimal script that demonstrates launching accelerate
on multiple remote GPUs, and with automatic hardware environment and dependency setup for reproducibility. You can
easily customize the training function used, training arguments, hyperparameters, and type of compute hardware, and then
run the script to automatically launch multi GPU training on remote hardware.

This script uses [Runhouse](https://github.com/run-house/runhouse) to launch on self-hosted hardware (e.g. in your own
cloud account or on-premise cluster) but there are other options for running remotely as well. Runhouse can be installed
with `pip install runhouse`, and you can refer to
[hardware setup](https://runhouse-docs.readthedocs-hosted.com/en/latest/api/python/cluster.html#hardware-setup)
for hardware setup instructions, or this
[Colab tutorial](https://colab.research.google.com/drive/1qVwYyLTCPYPSdz9ZX7BZl9Qm0A3j7RJe) for a more in-depth walkthrough.

## Finer Examples

While the first two scripts are extremely barebones when it comes to what you can do with accelerate, more advanced features are documented in two other locations.

### `by_feature` examples

These scripts are *individual* examples highlighting one particular feature or use-case within Accelerate. They all stem from the [nlp_example.py](./nlp_example.py) script, and any changes or modifications is denoted with a `# New Code #` comment.

Read the README.md file located in the `by_feature` folder for more information.

### `complete_*` examples

These two scripts contain *every* single feature currently available in Accelerate in one place, as one giant script.

New arguments that can be passed include:

- `checkpointing_steps`, whether the various states should be saved at the end of every `n` steps, or `"epoch"` for each epoch. States are then saved to folders named `step_{n}` or `epoch_{n}`
- `resume_from_checkpoint`, should be used if you want to resume training off of a previous call to the script and passed a `checkpointing_steps` to it.
- `with_tracking`, should be used if you want to log the training run using all available experiment trackers in your environment. Currently supported trackers include TensorBoard, Weights and Biases, and CometML.
