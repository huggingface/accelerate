# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Originally by jiwooya1000, put together together by sayakpaul.
Documentation: https://huggingface.co/docs/diffusers/main/en/training/distributed_inference

Run:

accelerate launch distributed_image_generation.py --batch_size 8

# Enable memory optimizations for large models like SD3
accelerate launch distributed_image_generation.py --batch_size 8 --low_mem
"""
import os
import time

import fire
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from tqdm import tqdm

from accelerate import PartialState
from accelerate.utils import gather_object


START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches


def main(
    ckpt_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    save_dir: str = "./evaluation/examples",
    seed: int = 1,
    batch_size: int = 4,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    dtype: str = "fp16",
    low_mem: bool = False,
):
    pipeline = DiffusionPipeline.from_pretrained(ckpt_id, torch_dtype=DTYPE_MAP[dtype])

    save_dir = save_dir + f"_{START_TIME}"

    parti_prompts = load_dataset("nateraw/parti-prompts", split="train")
    data_loader = get_batches(items=parti_prompts["Prompt"], batch_size=batch_size)

    distributed_state = PartialState()
    if low_mem:
        pipeline.enable_model_cpu_offload(gpu_id=distributed_state.device.index)
    else:
        pipeline = pipeline.to(distributed_state.device)

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    count = 0
    for _, prompts_raw in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_prompts = []

        with distributed_state.split_between_processes(prompts_raw) as prompts:
            generator = torch.manual_seed(seed)
            images = pipeline(
                prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator
            ).images
            input_prompts.extend(prompts)

        distributed_state.wait_for_everyone()

        images = gather_object(images)
        input_prompts = gather_object(input_prompts)

        if distributed_state.is_main_process:
            for image, prompt in zip(images, input_prompts):
                count += 1
                temp_dir = os.path.join(save_dir, f"example_{count}")

                os.makedirs(temp_dir)
                prompt = "_".join(prompt.split())
                image.save(f"image_{prompt}.png")

    if distributed_state.is_main_process:
        print(f">>> Image Generation Finished. Saved in {save_dir}")


if __name__ == "__main__":
    fire.Fire(main)
