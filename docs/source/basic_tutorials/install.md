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

# Installation

Before you start, you will need to setup your environment, install the appropriate packages, and configure Accelerate. Accelerate is tested on **Python 3.8+**.

Accelerate is available on pypi and conda, as well as on GitHub. Details to install from each are below:

## pip

To install Accelerate from pypi, perform:

```bash
pip install accelerate
```

## conda

Accelerate can also be installed with conda with:

```bash
conda install -c conda-forge accelerate
```

## Source

New features are added every day that haven't been released yet. To try them out yourself, install
from the GitHub repository:

```bash
pip install git+https://github.com/huggingface/accelerate
```

If you're working on contributing to the library or wish to play with the source code and see live 
results as you run the code, an editable version can be installed from a locally-cloned version of the 
repository:

```bash
git clone https://github.com/huggingface/accelerate
cd accelerate
pip install -e .
```

## Configuration

After installing, you need to configure Accelerate for how the current system is setup for training. 
To do so run the following and answer the questions prompted to you:

```bash
accelerate config
```

To write a barebones configuration that doesn't include options such as DeepSpeed configuration or running on TPUs, you can quickly run:

```bash
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```

Accelerate will automatically utilize the maximum number of GPUs available and set the mixed precision mode.

To check that your configuration looks fine, run:

```bash
accelerate env
```

An example output is shown below, which describes two GPUs on a single machine with no mixed precision being used:


```bash
- `Accelerate` version: 1.2.0.dev0
- Platform: Linux-6.8.0-47-generic-x86_64-with-glibc2.35
- `accelerate` bash location: /home/zach/miniconda3/envs/accelerate/bin/accelerate
- Python version: 3.10.13
- Numpy version: 1.26.4
- PyTorch version (GPU?): 2.5.1+cu124 (True)
- PyTorch XPU available: False
- PyTorch NPU available: False
- PyTorch MLU available: False
- PyTorch MUSA available: False
- System RAM: 187.91 GB
- GPU type: NVIDIA GeForce RTX 4090
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - debug: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - gpu_ids: all
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - enable_cpu_affinity: False
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
        - tpu_env: []
```
