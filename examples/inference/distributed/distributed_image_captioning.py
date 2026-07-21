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
Distributed Image Captioning using Hugging Face Accelerate.

This script demonstrates multi-GPU distributed inference for synthetic image captioning,
where each process runs a model copy on its assigned GPU and batches are split across devices.
Captions and metadata are saved asynchronously via a background thread to prevent GPU execution stalls.

Requirements:
    pip install transformers accelerate fire datasets pillow

Example usage:
    accelerate launch examples/inference/distributed/distributed_image_captioning.py --output_path captions_output --batch_size 8 --num_workers 2
"""

import json
import os
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import fire
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

from accelerate import PartialState
from accelerate.utils import tqdm


class ExistsFilter:
    """Filter out images whose caption metadata has already been saved."""

    def __init__(self, output_dir: Union[pathlib.Path, str]):
        current_files = [f.split("_metadata.json")[0] for f in os.listdir(output_dir) if f.endswith("_metadata.json")]
        self.processed_files = set(current_files)

    def __call__(self, sample):
        return sample["id"] not in self.processed_files


def load_caption_dataset(dataset_name: str = "lambdalabs/pokemon-blip-captions", split: str = "train", max_samples: Optional[int] = None):
    """Load image dataset with standard identifier field."""
    ds = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    dataset = []
    for idx, item in enumerate(ds):
        dataset.append(
            {
                "id": f"image_{idx:06d}",
                "image": item["image"],
            }
        )
    return dataset


def create_batches(items: List, batch_size: int):
    """Chunk items into list of batches."""
    num_batches = (len(items) + batch_size - 1) // batch_size
    return [items[i * batch_size : min((i + 1) * batch_size, len(items))] for i in range(num_batches)]


def save_captions(output_queue: queue.Queue, output_dir: pathlib.Path, model_name: str):
    """Asynchronous worker function for writing generated captions and metadata to disk."""
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break
            generated_texts, ids = item

            for caption, file_id in zip(generated_texts, ids):
                metadata = {
                    "id": file_id,
                    "caption": caption,
                    "model": model_name,
                }
                metadata_path = output_dir / f"{file_id}_metadata.json"
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)

        except queue.Empty:
            continue


def main(
    output_path: str = "captions_data",
    model_name: str = "Salesforce/blip-image-captioning-base",
    dataset_name: str = "lambdalabs/pokemon-blip-captions",
    dataset_split: str = "train",
    batch_size: int = 8,
    num_workers: int = 2,
    max_new_tokens: int = 50,
    dtype: str = "fp16",
):
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    distributed_state = PartialState()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model = model.to(distributed_state.device)
    model.eval()

    # Load dataset
    dataset = load_caption_dataset(dataset_name=dataset_name, split=dataset_split)
    exist_filter = ExistsFilter(output_dir)
    dataset = [item for item in dataset if exist_filter(item)]

    distributed_state.print(f"Total samples to caption across all processes: {len(dataset)}")

    # Split dataset across processes
    with distributed_state.split_between_processes(dataset) as local_dataset:
        local_batches = create_batches(local_dataset, batch_size)

        # Output queue and async save thread
        output_queue = queue.Queue()
        save_thread = ThreadPoolExecutor(max_workers=num_workers)
        save_future = save_thread.submit(save_captions, output_queue, output_dir, model_name)

        try:
            for batch in tqdm(local_batches, desc=f"Process {distributed_state.process_index} Captioning"):
                images = [sample["image"].convert("RGB") if isinstance(sample["image"], Image.Image) else sample["image"] for sample in batch]
                ids = [sample["id"] for sample in batch]

                inputs = processor(images=images, return_tensors="pt").to(distributed_state.device, dtype=torch_dtype)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_texts = [text.strip() for text in generated_texts]

                output_queue.put((generated_texts, ids))

        finally:
            output_queue.put(None)
            save_thread.shutdown(wait=True)

        save_future.result()

    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        distributed_state.print(f"Finished distributed image captioning. Results saved in {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
