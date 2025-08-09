# FP8 Benchmarks

Comparing and running [torchao](https://github.com/pytorch/ao/tree/main/torchao/float8) FP8 with accelerate

## Overview

This repo provides scripts which compare native `torchao` model training against `accelerate`'s own integration. Each modeling type is segmented out via a script, supporting the following:

* Single GPU training (`non_distributed.py`)
* Multi-GPU training via DistributedDataParallelism (`ddp.py`)
* Fully Sharded Data Parallelism (`fsdp.py`)
* DeepSpeed ZeRO 1-3 (`deepspeed.py`)

To run them, it's recommended to use a docker image (see the attached `Dockerfile`) and not install `torchao` manually.

## Running:

There are official Docker images located at `huggingface/accelerate:gpu-fp8-torchao-nightly` which can be used.

You can run all scripts using the core `accelerate launch` command without any `accelerate config` being needed.

For single GPU, run it via `python`:

```bash
python non_distributed.py
```

For the rest, run it via `accelerate launch`:

```bash
accelerate launch ddp.py # or distrib_deepspeed.py, ddp.py
```