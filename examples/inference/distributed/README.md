# Distributed inference examples

This folder contains a variety of tutorials for running distributed inference with the following strategy: 

Load an entire model onto each GPU and sending chunks of a batch through each GPU’s model copy at a time

## Installation

```bash
pip install accelerate torch
```

## Running code

You can either use `torchrun` or the recommended way of `accelerate launch` (without needing to run `accelerate config`) on each script:

```bash
accelerate launch --num_processes {NUM_GPUS} phi2.py
```

Or:

```bash
torchrun --nproc-per-node {NUM_GPUS} phi2.py
```

## Notes

You can also refer to [this gist](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba) that provides a more end-to-end example for image generation with advanced features like model offloading. You'd want to use `PartialState()` instead of `Accelerator()` there. 
