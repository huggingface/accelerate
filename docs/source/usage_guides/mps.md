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

# Accelerated PyTorch Training on Mac

With PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training. 
This unlocks the ability to perform machine learning workflows like prototyping and fine-tuning locally, right on Mac.
Apple's Metal Performance Shaders (MPS) as a backend for PyTorch enables this and can be used via the new `"mps"` device. 
This will map computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.
For more information please refer official documents [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
and [MPS BACKEND](https://pytorch.org/docs/stable/notes/mps.html).

### Benefits of Training and Inference using Apple Silicon Chips

1. Enables users to train larger networks or batch sizes locally
2. Reduces data retrieval latency and provides the GPU with direct access to the full memory store due to unified memory architecture. 
Therefore, improving end-to-end performance.
3. Reduces costs associated with cloud-based development or the need for additional local GPUs.

**Pre-requisites**: To install torch with mps support, 
please follow this nice medium article [GPU-Acceleration Comes to PyTorch on M1 Macs](https://medium.com/towards-data-science/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1).


## How it works out of the box
It is enabled by default on MacOs machines with MPS enabled Apple Silicon GPUs.
To disable it, pass `--cpu` flag to `accelerate launch` command or answer the corresponding question when answering the `accelerate config` questionnaire.

You can directly run the following script to test it out on MPS enabled Apple Silicon machines:
```bash
accelerate launch /examples/cv_example.py --data_dir images
```

## A few caveats to be aware of

1. We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing) on your MacOS machine. 
It has major fixes related to model correctness and performance improvements for transformer based models.
Please refer to https://github.com/pytorch/pytorch/issues/82707 for more details.
2. Distributed setups `gloo` and `nccl` are not working with `mps` device. 
This means that currently only single GPU of `mps` device type can be used.

Finally, please, remember that, 🤗 `Accelerate` only integrates MPS backend, therefore if you
have any problems or questions with regards to MPS backend usage, please, file an issue with [PyTorch GitHub](https://github.com/pytorch/pytorch/issues).