# FP8 Benchmarks

Comparing and running [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) FP8 with accelerate

## Overview

This repo provides scripts which compare native TransformerEngine model training against `accelerate`'s own integration. Each modeling type is segmented out via a script, supporting the following:

* Single GPU training (`non_distributed.py`)
* Multi-GPU training via DistributedDataParallelism (`ddp.py`)
* Fully Sharded Data Parallelism (`fsdp.py`)
* DeepSpeed ZeRO 1-3 (`deepspeed.py`)

NVIDIA Transformer Engine >= 2.9.0 is required. The attached `Dockerfile` defaults to NVIDIA PyTorch `26.09-py3`
and TE `2.20.2`, and installs your checked-out Accelerate source. Build it from the repository root:

```bash
docker build -f benchmarks/fp8/transformer_engine/Dockerfile -t accelerate-te-benchmarks .
docker run --gpus all --ipc=host --rm -it accelerate-te-benchmarks
```

## Running:

Inside the image, the working directory is `benchmarks/fp8/transformer_engine`.

You can run all scripts using the core `accelerate launch` command without any `accelerate config` being needed.

For single GPU, run it via `python`:

```bash
python non_distributed.py
```

For the rest, run it via `accelerate launch`:

```bash
accelerate launch ddp.py # or distrib_deepspeed.py, ddp.py
```
