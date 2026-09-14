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
Distributed image captioning example using BLIP-2 with Accelerate.

This example demonstrates how to run distributed image captioning on multiple GPUs
using the `PartialState` API for simple data-parallel inference.

Run with:
    accelerate launch --num_processes {NUM_GPUS} image_captioning.py

Or:
    torchrun --nproc-per-node {NUM_GPUS} image_captioning.py

Requirements:
    pip install transformers accelerate torch Pillow
"""

import json
import os
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import fire
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from accelerate import PartialState
from accelerate.utils import tqdm


def main(
    data_path: str,
    output_path: str,
    batch_size: int = 8,
    num_workers: int = 2,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    max_new_tokens: int = 50,
    num_beams: int = 3,
):
    """
    Distributed image captioning with BLIP-2.

    Args:
        data_path: Path to directory containing images or a single image file
        output_path: Output directory for captions and metadata
        batch_size: Batch size per GPU
        num_workers: Number of workers for saving results
        model_name: BLIP-2 model to use
        max_new_tokens: Maximum new tokens for caption generation
        num_beams: Number of beams for beam search
    """
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    distributed_state = PartialState()

    # Load model and processor
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map=distributed_state.device,
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    def find_images(root: Union[str, pathlib.Path]):
        """Find all image files in directory recursively."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        root = pathlib.Path(root)
        if root.is_file() and root.suffix.lower() in image_extensions:
            return [root]
        return [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

    image_files = find_images(data_path)
    if not image_files:
        raise ValueError(f"No image files found in {data_path}")

    # Filter already processed images
    processed_hashes = set()
    if output_dir.exists():
        for f in output_dir.glob("*_caption.json"):
            processed_hashes.add(f.stem.replace("_caption", ""))

    def get_file_hash(path: pathlib.Path) -> str:
        """Generate a hash for the file path to use as unique identifier."""
        import hashlib
        return hashlib.md5(str(path).encode()).hexdigest()[:12]

    # Filter out already processed
    image_files = [
        img for img in image_files
        if get_file_hash(img) not in processed_hashes
    ]

    distributed_state.print(f"Found {len(image_files)} images to process")

    if not image_files:
        distributed_state.print("All images already processed!")
        return

    def preprocess_batch(batch_images, processor):
        """Preprocess a batch of images."""
        images = [Image.open(img).convert("RGB") for img in batch_images]
        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return inputs, images

    def save_results(output_queue: queue.Queue, output_dir: pathlib.Path, processor):
        """Save results in a separate thread to not block GPU."""
        while True:
            try:
                item = output_queue.get(timeout=5)
                if item is None:
                    break
                captions, image_paths, images = item
                for caption, img_path, image in zip(captions, image_paths, images):
                    file_hash = get_file_hash(img_path)

                    # Save image
                    img_save_path = output_dir / f"{file_hash}.jpg"
                    image.save(img_save_path)

                    # Save caption metadata
                    metadata = {
                        "original_path": str(img_path),
                        "caption": caption,
                        "model": model_name,
                        "max_new_tokens": max_new_tokens,
                        "num_beams": num_beams,
                    }
                    meta_path = output_dir / f"{file_hash}_caption.json"
                    with meta_path.open("w") as f:
                        json.dump(metadata, f, indent=2)
            except queue.Empty:
                continue

    # Create batches
    batches = [
        image_files[i:i + batch_size]
        for i in range(0, len(image_files), batch_size)
    ]

    # Distribute batches across processes
    if distributed_state.num_processes > 1:
        chunk_size = len(batches) // distributed_state.num_processes
        start_idx = distributed_state.process_index * chunk_size
        end_idx = (
            start_idx + chunk_size
            if distributed_state.process_index < distributed_state.num_processes - 1
            else len(batches)
        )
        batches = batches[start_idx:end_idx]

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_workers)
    save_future = save_thread.submit(save_results, output_queue, output_dir, processor)

    try:
        for batch_images in tqdm(
            batches,
            desc="Generating captions",
            disable=not distributed_state.is_main_process,
        ):
            inputs, images = preprocess_batch(batch_images, processor)
            inputs = {k: v.to(distributed_state.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

            captions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            output_queue.put((captions, batch_images, images))

    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    distributed_state.print(f"\n✅ Captioning completed! Results saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)