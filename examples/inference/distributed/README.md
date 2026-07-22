# Distributed Inference Examples

This folder contains a variety of examples for running distributed inference with Hugging Face `accelerate`.

## Strategy

Load an entire copy of the model onto each available GPU and process split chunks of a dataset batch across GPUs concurrently. Artifact serialization (saving audio/captions/metadata) is handled in non-blocking background threads.

## Examples Included

- **`distributed_image_captioning.py`**: Distributed synthetic image captioning using BLIP models with async metadata serialization.
- **`distributed_speech_generation.py`**: Distributed speech synthesis (TTS) using VITS models with async WAV output serialization.
- **`distributed_image_generation.py`**: Distributed image generation with PixArt / SD3 pipelines and memory optimization support (`--low_mem`).
- **`stable_diffusion.py`**: Multi-GPU batch prompt generation using Stable Diffusion.
- **`florence2.py`**: Distributed vision-language inference using Florence-2.
- **`llava_next_video.py`**: Distributed video understanding using LLaVA-NeXT-Video.
- **`phi2.py`**: Distributed text generation using Phi-2.

## Installation

```bash
pip install accelerate torch transformers datasets fire pillow scipy
```

## Running Examples

You can run scripts using `accelerate launch`:

```bash
# Distributed Image Captioning
accelerate launch examples/inference/distributed/distributed_image_captioning.py --batch_size 8 --num_workers 2

# Distributed Speech Generation
accelerate launch examples/inference/distributed/distributed_speech_generation.py --output_path outputs --batch_size 8

# Distributed Image Generation
accelerate launch examples/inference/distributed/distributed_image_generation.py --batch_size 8 --low_mem
```

Or using `torchrun`:

```bash
torchrun --nproc-per-node {NUM_GPUS} examples/inference/distributed/distributed_image_captioning.py
```

