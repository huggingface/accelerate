<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Intel Gaudi

Users can take advantage of Intel Gaudi AI accelerators for significantly faster and cost-effective model training and inference.
The Intel Gaudi AI accelerator family currently includes three product generations: [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), and [Intel Gaudi 3](https://habana.ai/products/gaudi3/). Each server is equipped with 8 devices, known as Habana Processing Units (HPUs), providing 128GB of memory on Gaudi 3, 96GB on Gaudi 2, and 32GB on the first-gen Gaudi. For more details on the underlying hardware architecture, check out the [Gaudi Architecture Overview](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html).

## How it works out of the box

It is enabled by default if an Intel Gaudi device is detected.
To disable it, pass `--cpu` flag to `accelerate launch` command or answer the corresponding question when answering the `accelerate config` questionnaire.

You can directly run the following script to test it out on Intel Gaudi:

```bash
accelerate launch /examples/cv_example.py --data_dir images
```

## Limitations

The following features are not part of the Accelerate library and requires [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index):

- `fast_ddp` which implements DDP by applying an all-reduce on gradients instead of the Torch DDP wrapper.
- `minimize_memory` which is used for fp8 training and enables keeping fp8 weights in memory between the forward and backward passes, leading to a smaller memory footprint at the cost of additional fp8 casts.
- `context_parallel_size` which is used for Context/Sequence Parallelism (CP/SP) and partitions the network inputs and activations along sequence dimension to reduce memory footprint and increase throughput.
