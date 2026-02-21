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

## Available examples

These scripts all use the same high-level strategy: each process loads a full copy of the model on its device, and
`PartialState.split_between_processes` is used to split work across ranks.

- **`phi2.py`**: text generation with an LLM (Phi-2).
- **`stable_diffusion.py` / `distributed_image_generation.py`**: image generation with Diffusers.
- **`florence2.py`**: image captioning with Florence-2 (supports streaming WebDataset inputs and threaded result saving).
- **`distributed_speech_generation.py`**: TTS / speech generation example with threaded audio + metadata saving.
- **`llava_next_video.py`**: multimodal / video inference example.

Most of the examples support an `--output_path` to write artifacts (images/audio/JSON) to disk. When relevant, the
writing is done on a background thread to avoid blocking GPU compute.
