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
Distributed image captioning example using accelerate.

Demonstrates how to split batches of image-caption pairs across multiple GPUs
and generate captions for each image using a BLIP-2 or similar vision-language model.

Run:

accelerate launch distributed_image_captioning.py --batch_size 8

Or with a different model and dataset:

accelerate launch distributed_image_captioning.py \
    --model_name "Salesforce/blip2-opt-2.7b" \
    --dataset_name "ydshieh/coco_dataset_script" \
    --dataset_config "2017" \
    --batch_size 8 \
    --max_new_tokens 30
"""

import json
import os
import time

import fire
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from accelerate import PartialState
from accelerate.utils import gather_object


START_TIME = time.strftime("%Y%m%d_%H%M%S")


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
    model_name: str = "Salesforce/blip2-opt-2.7b",
    dataset_name: str = "nlphuji/flickr30k",
    dataset_config: str = None,
    dataset_split: str = "test",
    save_dir: str = "./captioning_results",
    seed: int = 42,
    batch_size: int = 4,
    max_new_tokens: int = 30,
    num_beams: int = 5,
    dtype: str = "fp16",
):
    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = DTYPE_MAP[dtype]

    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    save_dir = save_dir + f"_{START_TIME}"
    distributed_state = PartialState()

    model = model.to(distributed_state.device)

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    data_loader = get_batches(items=list(range(len(dataset))), batch_size=batch_size)

    if distributed_state.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to {save_dir}")

    results_accumulator = []
    for batch_indices in tqdm(data_loader, disable=not distributed_state.is_main_process):
        with distributed_state.split_between_processes(batch_indices) as local_indices:
            local_images = []
            for idx in local_indices:
                example = dataset[int(idx)]
                image = example["image"]
                if isinstance(image, Image.Image):
                    local_images.append(image.convert("RGB"))
                else:
                    local_images.append(image)

            pixel_values = processor(
                images=local_images, return_tensors="pt"
            ).pixel_values.to(distributed_state.device, dtype=torch_dtype)

            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            generated_captions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        distributed_state.wait_for_everyone()

        indices_gather = gather_object(local_indices)
        captions_gather = gather_object(generated_captions)
        if distributed_state.is_main_process:
            for idx, caption in zip(indices_gather, captions_gather):
                results_accumulator.append({"index": int(idx), "caption": caption})

    if distributed_state.is_main_process:
        results_path = os.path.join(save_dir, "captions.json")
        with open(results_path, "w") as f:
            json.dump(results_accumulator, f, indent=2)

        for entry in results_accumulator:
            idx = entry["index"]
            example = dataset[int(idx)]
            image = example["image"]
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            image.save(os.path.join(save_dir, f"image_{idx}.jpg"))

        print(f">>> Captioning finished. {len(results_accumulator)} images processed.")
        print(f">>> Results saved to {save_dir}")


if __name__ == "__main__":
    fire.Fire(main)
