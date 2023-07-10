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

# Installation and Configuration

Before you start, you will need to setup your environment, install the appropriate packages, and configure 🤗 Accelerate. 🤗 Accelerate is tested on **Python 3.8+**.

## Installing 🤗 Accelerate

🤗 Accelerate is available on pypi and conda, as well as on GitHub. Details to install from each are below:

### pip 

To install 🤗 Accelerate from pypi, perform:

```bash
pip install accelerate
```

### conda

🤗 Accelerate can also be installed with conda with:

```bash
conda install -c conda-forge accelerate
```

### Source

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

## Configuring 🤗 Accelerate

After installing, you need to configure 🤗 Accelerate for how the current system is setup for training. 
To do so run the following and answer the questions prompted to you:

```bash
accelerate config
```

To write a barebones configuration that doesn't include options such as DeepSpeed configuration or running on TPUs, you can quickly run:

```bash
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```
🤗 Accelerate will automatically utilize the maximum number of GPUs available and set the mixed precision mode.

To check that your configuration looks fine, run:

```bash
accelerate env
```

An example output is shown below, which describes two GPUs on a single machine with no mixed precision being used:

```bash
- `Accelerate` version: 0.11.0.dev0
- Platform: Linux-5.10.0-15-cloud-amd64-x86_64-with-debian-11.3
- Python version: 3.7.12
- Numpy version: 1.19.5
- PyTorch version (GPU?): 1.12.0+cu102 (True)
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - main_process_ip: None
        - main_process_port: None
        - main_training_function: main
        - deepspeed_config: {}
        - fsdp_config: {}
```