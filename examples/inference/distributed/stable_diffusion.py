import torch
from diffusers import DiffusionPipeline

from accelerate import PartialState  # Can also be Accelerator or AcceleratorState


pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipe.to(distributed_state.device)

# Assume two processes
with distributed_state.split_between_processes(["a dog", "a cat", "a chicken"], apply_padding=True) as prompt:
    result = pipe(prompt).images
