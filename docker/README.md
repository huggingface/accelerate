<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

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

# Official Hugging Face Accelerate Docker Images

Accelerate publishes a variety of docker versions as part of our CI that users can also use. These are stable images that Accelerate can run off of which comes with a variety of different setup configurations, all of which are officially hosted on [Docker Hub](https://hub.docker.com/r/huggingface/accelerate).

A breakdown of each are given below

## Naming Conventions

Accelerate docker images follow a tagging convention of:

```bash
huggingface/accelerate:{accelerator}-{nightly,release}
```

`accelerator` in this instance is one of many applical pre-configured backend supports:
* `gpu`: Comes compiled off of the `nvidia/cuda` image and includes core parts like `bitsandbytes`. Runs off python 3.9.
* `cpu`: Comes compiled off of `python:3.9-slim` and is designed for non-CUDA based workloads.
* More to come soon
* `gpu-deepspeed`: Comes compiled off of the `nvidia/cuda` image and includes core parts like `bitsandbytes` as well as the latest `deepspeed` version. Runs off python 3.10.

## Nightlies vs Releases

Each release a new build is pushed with a version number included in the name. For a GPU-supported image of version 0.28.0 for instance, it would look like the following:

```bash
huggingface/accelerate:gpu-release-0.28.0
```

Nightlies contain two different image tags. There is a general `nightly` tag which is built each night, and a `nightly-YYYY-MM-DD` which corresponds to a build from a particular date.

For instance, here is an example nightly CPU image from 3/14/2024

```bash
huggingface/accelerate:cpu-nightly-2024-03-14
```

## Running the images

Each image comes compiled with `conda` and an `accelerate` environment contains all of the installed dependencies. 

To pull down the latest nightly run:

```bash
docker pull huggingface/accelerate:gpu-nightly
```

To then run it in interactive mode with GPU-memory available, run:

```bash
docker container run --gpus all -it huggingface/accelerate:gpu-nightly
```

## DEPRECATED IMAGES

CPU and GPU docker images were hosted at `huggingface/accelerate-gpu` and `huggingface/accelerate-cpu`. These builds are now outdated and will not receive updates. 

The builds at the corresponding `huggingface/accelerate:{gpu,cpu}` contain the same `Dockerfile`, so it's as simple as changing the docker image to the desired ones from above. We will not be deleting these images for posterity, but they will not be receiving updates going forward.