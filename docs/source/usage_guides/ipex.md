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

# Intel® Extension for PyTorch

[IPEX](https://github.com/intel/intel-extension-for-pytorch) is optimized for CPUs with AVX-512 or above, and functionally works for CPUs with only AVX2. So, it is expected to bring performance benefit for Intel CPU generations with AVX-512 or above while CPUs with only AVX2 (e.g., AMD CPUs or older Intel CPUs) might result in a better performance under IPEX, but not guaranteed. IPEX provides performance optimizations for CPU training with both Float32 and BFloat16. The usage of BFloat16 is the main focus of the following sections.

Low precision data type BFloat16 has been natively supported on the 3rd Generation Xeon® Scalable Processors (aka Cooper Lake) with AVX512 instruction set and will be supported on the next generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix Extensions (Intel® AMX) instruction set with further boosted performance. The Auto Mixed Precision for CPU backend has been enabled since PyTorch-1.10. At the same time, the support of Auto Mixed Precision with BFloat16 for CPU and BFloat16 optimization of operators has been massively enabled in Intel® Extension for PyTorch, and partially upstreamed to PyTorch master branch. Users can get better performance and user experience with IPEX Auto Mixed Precision.

## IPEX installation:

IPEX release is following PyTorch, to install via pip:

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 2.0               |  2.0.0         |
| 1.13              |  1.13.0        |
| 1.12              |  1.12.300      |
| 1.11              |  1.11.200      |
| 1.10              |  1.10.100      |

```
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

Check more approaches for [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).


## How It Works For Training optimization in CPU

Accelerate has integrated [IPEX](https://github.com/intel/intel-extension-for-pytorch), all you need to do is enabling it through the config.

**Scenario 1**: Acceleration of No distributed CPU training

Run <u>accelerate config</u> on your machine:

```bash
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon device is available)? [yes/NO]:yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: NO
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```
This will generate a config file that will be used automatically to properly set the
default options when doing

```bash
accelerate launch my_script.py --args_to_my_script
```

For instance, here is how you would run the NLP example `examples/nlp_example.py` (from the root of the repo) with IPEX enabled.
default_config.yaml that is generated after `accelerate config`

```bash
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
ipex_config:
  ipex: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```
```bash
accelerate launch examples/nlp_example.py
```

**Scenario 2**: Acceleration of distributed CPU training
we use Intel oneCCL for communication, combined with Intel® MPI library to deliver flexible, efficient, scalable cluster messaging on Intel® architecture. you could refer the [here](https://huggingface.co/docs/transformers/perf_train_cpu_many) for the installation guide

Run <u>accelerate config</u> on your machine(node0):

```bash
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-CPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 4
-----------------------------------------------------------------------------------------------------------------------------------------------------------
What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? 36.112.23.24
What is the port you will use to communicate with the main process? 29500
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you want accelerate to launch mpirun? [yes/NO]: yes
Please enter the path to the hostfile to use with mpirun [~/hostfile]: ~/hostfile
Enter the number of oneCCL worker threads [1]: 1
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
How many processes should be used for distributed training? [1]:16
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```
For instance, here is how you would run the NLP example `examples/nlp_example.py` (from the root of the repo) with IPEX enabled for distributed CPU training.

default_config.yaml that is generated after `accelerate config`
```bash
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_CPU
downcast_bf16: 'no'
ipex_config:
  ipex: true
machine_rank: 0
main_process_ip: 36.112.23.24
main_process_port: 29500
main_training_function: main
mixed_precision: bf16
mpirun_config:
  mpirun_ccl: '1'
  mpirun_hostfile: /home/user/hostfile
num_machines: 4
num_processes: 16
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
```

Set following env and using intel MPI to launch the training

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
If you selected to have Accelerate launch `mpirun`, ensure that the location of your hostfile matches the path in the config.
```bash
$ cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
xxx.xxx.xxx.xxx #node2 ip
xxx.xxx.xxx.xxx #node3 ip
```
When Accelerate is launching `mpirun`, source the oneCCL bindings setvars.sh to get your Intel MPI environment, and then
run your script using `accelerate launch`. Note that the python script and environment needs to exist on all of the
machines being used for multi-CPU training.
```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

accelerate launch examples/nlp_example.py
```
Otherwise, if you selected not to have Accelerate launch `mpirun`, run the following command in node0 and **16DDP** will
be enabled in node0,node1,node2,node3 with BF16 mixed precision. When using this method, the python script, python
environment, and accelerate config file need to be present on all of the machines used for multi-CPU training.
```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
export CCL_WORKER_COUNT=1
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
export CCL_ATL_TRANSPORT=ofi
mpirun -f hostfile -n 16 -ppn 4 accelerate launch examples/nlp_example.py
```

## Related Resources

- [Project's github](https://github.com/intel/intel-extension-for-pytorch)
- [API docs](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/api_doc.html)
- [Tuning guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html)
- [Blogs & Publications](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/blogs_publications.html)

