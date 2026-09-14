# Distributed inference examples

This folder contains a variety of tutorials for running distributed inference with the following strategy: 

Load an entire model onto each GPU and sending chunks of a batch through each GPU’s model copy at a time

## Installation

```bash
pip install accelerate torch
```

## Examples

| Script | Model | Description |
|--------|-------|-------------|
| `phi2.py` | Microsoft Phi-2 | Basic distributed inference with Phi-2 |
| `gemma_distributed.py` | Google Gemma 2B | Distributed inference with Gemma 2B-IT |
| `image_captioning.py` | BLIP-2 | Distributed image captioning |
| `stable_diffusion.py` | Stable Diffusion | Distributed image generation |
| `distributed_image_generation.py` | Various | Image generation examples |
| `distributed_speech_generation.py` | Various | Speech generation examples |
| `florence2.py` | Florence-2 | Vision-language model inference |
| `llava_next_video.py` | LLaVA-NeXT-Video | Video understanding |

## Running code

You can either use `torchrun` or the recommended way of `accelerate launch` (without needing to run `accelerate config`) on each script:

```bash
accelerate launch --num_processes {NUM_GPUS} phi2.py
```

Or:

```bash
torchrun --nproc-per-node {NUM_GPUS} phi2.py
```

For the Gemma example:

```bash
accelerate launch --num_processes {NUM_GPUS} gemma_distributed.py
```

For the image captioning example:

```bash
accelerate launch --num_processes {NUM_GPUS} image_captioning.py --data_path /path/to/images --output_path /path/to/output
```
