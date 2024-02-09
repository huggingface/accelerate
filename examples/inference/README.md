# Distributed inference examples with PiPPy

This repo contains a variety of tutorials for using the [PiPPy](https://github.com/PyTorch/PiPPy) pipeline parallelism library with accelerate. You will find examples covering:

1. How to trace the model using `accelerate.prepare_pippy`
2. How to specify inputs based on what the model expects (when to use `kwargs`, `args`, and such)
3. How to gather the results at the end.

## Installation

This requires the `main` branch of accelerate (or a version at least 0.27.0),  `pippy` version of 0.2.0 or greater, and at least python 3.9. Please install using `pip install .` to pull from the `setup.py` in this repo, or run manually:

```bash
pip install 'accelerate>=0.27.0' 'torchpippy>=0.2.0'
```

## Running code

You can either use `torchrun` or the recommended way of `accelerate launch` (without needing to run `accelerate config`) on each script:

```bash
accelerate launch bert.py
```

Or:

```bash
accelerate launch --num_processes {NUM_GPUS} bert.py
```

Or:

```bash
torchrun --nproc-per-node {NUM_GPUS} bert.py
```

## General speedups

One can expect that PiPPy will outperform native model parallism by a multiplicative factor since all GPUs are running at all times with inputs, rather than one input being passed through a GPU at a time waiting for the prior to finish. 

Below are some benchmarks we have found when using the accelerate-pippy integration for a few models when running on 2x4090's:

### Bert

|  | Accelerate/Sequential | PiPPy + Accelerate |
|---|---|---|
| First batch | 0.2137s | 0.3119s |
| Average of 5 batches | 0.0099s | **0.0062s** |

### GPT2

|  | Accelerate/Sequential | PiPPy + Accelerate |
|---|---|---|
| First batch | 0.1959s | 0.4189s |
| Average of 5 batches | 0.0205s | **0.0126s** |

### T5

|  | Accelerate/Sequential | PiPPy + Accelerate |
|---|---|---|
| First batch | 0.2789s | 0.3809s |
| Average of 5 batches | 0.0198s | **0.0166s** |