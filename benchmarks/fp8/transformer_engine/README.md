# FP8 Benchmarks

Comparing and running [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) FP8 with accelerate

## Overview

This repo provides scripts which compare native TransformerEngine model training against `accelerate`'s own integration. Each modeling type is segmented out via a script, supporting the following:

* Single GPU training (`non_distributed.py`)
* Multi-GPU training via DistributedDataParallelism (`ddp.py`)
* Fully Sharded Data Parallelism (`fsdp.py`)
* DeepSpeed ZeRO 1-3 (`distrib_deepspeed.py`)

The parity scripts compare native TE FP8 training with Accelerate TE FP8 training. They check that both paths produce
the same accuracy and F1, rather than requiring FP8 to improve accuracy over an untrained model.

`performance.py` separately measures single-GPU training with PyTorch BF16, native TE BF16, native TE FP8, and
Accelerate TE FP8. It reports synchronized step time, samples/second, padded tokens/second, and peak allocated/reserved
CUDA memory. Each case runs in a fresh process with the same seed, data, optimizer, and step budget. Warmup steps,
model/data downloads, initialization, and evaluation are excluded from timing. The default is three repetitions;
the JSON report contains every run and the median performance for each case.

Accuracy and F1 are evaluated in BF16 on all 408 MRPC validation examples before and after the training steps.
The report includes each case's metric differences from BF16. These are short training runs from a base BERT model,
not a convergence study. FP8 should preserve model quality within a tolerance established by recipe validation;
the reported deltas help identify regressions but do not establish that tolerance or prove quality preservation.
Unchanged accuracy is not a performance failure. FP8 speedups also depend on model and GPU size, so the benchmark
does not require FP8 to be faster.

Use the attached Dockerfile, which defaults to NVIDIA's `26.09-py3` PyTorch container and upgrades TE to `2.20.2`. Build it
from the repository root so the image tests your checked-out Accelerate source:

```bash
docker build -f benchmarks/fp8/transformer_engine/Dockerfile -t accelerate-te-benchmarks .
docker run --gpus all --ipc=host --rm -it accelerate-te-benchmarks
```

For the single-GPU performance benchmark, expose exactly one FP8-capable GPU (for example, an L4). Driver
requirements follow the selected NVIDIA container. The base image can be overridden with the `BASE_YEAR` and
`BASE_MONTH` build arguments. `TE_VERSION` selects the TE release. The PyTorch extension is built with two workers;
the build requires CUDA development headers and enough host RAM for compilation.

## Running:

Inside the image, the working directory is `benchmarks/fp8/transformer_engine`.

Run the performance comparison and save its JSON report:

```bash
python performance.py --warmup-steps 10 --steps 100 --repeats 3 --output performance.json
```

Use `--model-name` and `--batch-size` to change the workload. Tokens/second includes padding because the GEMMs process
the padded sequences. `speedup_vs_bf16` compares each case to PyTorch BF16; `speedup_vs_te_bf16` isolates the effect of
FP8 on native TE layers.

For a causal-LM workload such as Qwen2.5-7B, use packed WikiText sequences:

```bash
python performance.py --task causal-lm --model-name Qwen/Qwen2.5-7B \
  --batch-size 1 --sequence-length 1024 --warmup-steps 10 --steps 50 \
  --repeats 3 --output qwen7b-performance.json
```

This mode trains all parameters with FP32 weights/gradients and fused AdamW with FP32 moments, while matrix
multiplications use BF16 or TE FP8. It reports held-out loss, perplexity, and next-token accuracy before/after training
on eight fixed validation sequences, evaluated in BF16. All cases use the same packed batches, SDPA attention,
learning rate (default `1e-5`), and optimizer. `--eval-sequences`, `--sequence-length`, and `--learning-rate` are configurable.
No attention fusion or optimizer precision changes are added only to FP8 cases.

Qwen2.5-7B's 7.61B parameters require approximately 122 GB for FP32 parameters, gradients, and AdamW moments alone;
activations, quantization buffers, and CUDA workspaces need additional memory. Start with batch size 1 on an H200
141 GB and check measured memory before increasing the batch or sequence length. A single H100 80 GB cannot hold
this full optimizer setup. The short-run quality report is a smoke check, not convergence validation.

The distributed parity scripts require suitable launch configurations and multiple GPUs. They do not measure
distributed throughput or communication performance.

For single GPU, run it via `python`:

```bash
python non_distributed.py
```

For the rest, run it via `accelerate launch`:

```bash
accelerate launch --multi_gpu --num_processes 2 ddp.py
accelerate launch --use_fsdp --num_processes 2 fsdp.py
accelerate launch --use_deepspeed --num_processes 2 distrib_deepspeed.py
```
