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
from transformers import BlipForConditionalGeneration, BlipProcessor
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
    model_id: str = "Salesforce/blip-image-captioning-base",  # Choose your captioning model
    save_dir: str = "./evaluation/captions",
    seed: int = 1,
    batch_size: int = 4,
    dtype: str = "fp16",
):
    # Load processor and model for captioning
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(dtype=DTYPE_MAP[dtype])

    save_dir = save_dir + f"_{START_TIME}"

    # Load your image dataset; this is an example using a dummy dataset
    ds = load_dataset("uoft-cs/cifar10", split="train")
    data_loader = get_batches(items=ds["img"], batch_size=batch_size)  # Ensure your dataset has an 'image' key

    distributed_state = PartialState()
    
    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    count = 0
    for _, images_raw in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_images = []

        with distributed_state.split_between_processes(images_raw) as images:
            generator = torch.manual_seed(seed)
            # Process images and generate captions
            inputs = processor(images, return_tensors="pt", padding=True).to(dtype=DTYPE_MAP[dtype])
            outputs = model.generate(**inputs)
            captions = processor.batch_decode(outputs, skip_special_tokens=True)

            input_images.extend(images)

        distributed_state.wait_for_everyone()

        captions = gather_object(captions)
        input_images = gather_object(input_images)

        if distributed_state.is_main_process:
            for caption, img in zip(captions, input_images):
                count += 1
                temp_dir = os.path.join(save_dir, f"example_{count}")

                os.makedirs(temp_dir, exist_ok=True)
                img.save(os.path.join(temp_dir, f"image_{count}.png"))
                with open(os.path.join(temp_dir, "caption.txt"), "w") as f:
                    f.write(caption)

    if distributed_state.is_main_process:
        print(f">>> Caption Generation Finished. Saved in {save_dir}")

if __name__ == "__main__":
    fire.Fire(main)