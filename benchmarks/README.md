# Big model inference benchmarks

Running inference with Accelerate on big models.

## Setup

These benchmarks use the `transformers` library:

```bash
pip install transformers
```

To reproduce or test a new setup, run

```py
python inference_acc.py model_name
```

This script supports `gpt-j-6b`, `gpt-neox`, `opt` (30B version) and `T0pp` out of the box, but you can specify any valid checkpoint for `model_name`.

To force a different `torch_dtype` than the one in the config: `--torch_dtype xxx`.

If you get an error linked to disk offload, you need to add the option `--disk-offload`

## Results

On a setup with two Titan RTXs (24GB of RAM) and 32GB of RAM, we get the following benchmarks (T0pp does not run in float16, which is why it's not included).

| Model | Model load time | Generation time | dtype | GPU 0 use | GPU 1 use | CPU use | Disk offload |
|:-----:|:---------------:|:---------------:|:-----:|:---------:|:---------:|:-------:|:------------:|
| GPT-J-6B | 8.7s | 0.05s per token | float16 | 11.7GB | 0GB | 0GB | no |
| GPT-J-6B | 12.4s | 0.06s per token | float32 | 21.9GB | 1.5GB | 0GB | no |
| GPT-Neo-X-20B | 30.9s | 0.08s per token | float16 | 21.5GB | 18GB | 0GB | no |
| GPT-Neo-X-20B | 78.2s | 10.72s per token | float32 | 20.3GB | 22.7 GB | 24.4GB | yes |
| T0pp (11B) | 29.4s | 0.05s per token | float32 | 21.1GB | 21.3GB | 0GB | no |
| OPT-30B | 34.5s | 2.37s per token | float16 | 20.7GB | 22.3GB | 14.1GB | no |
| OPT-30B | 112.3s | 33.9s per token | float32 | 20.2GB | 21.2GB | 23.5GB | yes |

Note on the results:
- using two GPUs instead of one does not slow down generation
- using CPU offload slows down a bit (see OPT-30b)
- using disk offload slows down a lot (need to implement prefetching)

You will also note that Accelerate does not use anymore GPU and CPU RAM than necessary:
- peak GPU memory is exactly the size of the model put on a given GPU
- peak CPU memory is either the size of the biggest checkpoint shard or the part of the model offloaded on CPU, whichever is bigger.