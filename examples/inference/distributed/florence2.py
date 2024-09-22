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

import json
import os
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Union

import fire
import torch
import webdataset as wds
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from accelerate import PartialState


"""
Additional requirements: flash_attn einops timm
pip install flash_attn einops timm

Example:

accelerate launch --num_processes=2 florence2.py --data_path "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-0000.tar" --output_path outputs --batch_size 12 --num_workers 1 --prompt "<CAPTION>"

On 2x4090: 420it [03:15,  2.15it/s] (~25.8 images/s)

With --prompt "<DETAILED_CAPTION>": 420it [08:16,  1.18s/it] (~10.17 images/s)
"""


def main(
    data_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    prompt: str = "<MORE_DETAILED_CAPTION>",
    model_name: str = "microsoft/Florence-2-large",
    max_new_tokens: int = 1024,
    num_beams: int = 3,
):
    output_dir = pathlib.Path(output_path)

    distributed_state = PartialState()

    if distributed_state.is_main_process:
        output_dir.mkdir(exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=distributed_state.device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=True)

    class ExistsFilter:
        def __init__(self, output_dir: Union[pathlib.Path, str]):
            current_training_img_hashes = [f.split(".jpg")[0] for f in os.listdir(output_dir) if f.endswith(".jpg")]
            self.current_training_img_hashes = set(current_training_img_hashes)
            if distributed_state.is_main_process:
                print(f"Existing images found: {len(self.current_training_img_hashes)}.")

        def __call__(self, x):
            if len(self.current_training_img_hashes) > 0:
                if x["img_hash"] in self.current_training_img_hashes:
                    return False
                else:
                    return True
            else:
                return True

    def preprocess_fn(sample, processor):
        image: Image.Image = sample["jpg"].convert("RGB")
        img_hash = insecure_hashlib.sha1(image.tobytes()).hexdigest()
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "image": image,
            "img_hash": img_hash,
            "original_caption": sample["txt"],
        }

    def collate_fn(examples):
        input_ids = torch.cat([example["input_ids"] for example in examples])
        pixel_values = torch.cat([example["pixel_values"] for example in examples])
        images = [example["image"] for example in examples]
        img_hashes = [example["img_hash"] for example in examples]
        captions = [example["original_caption"] for example in examples]
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "images": images,
            "img_hashes": img_hashes,
            "original_captions": captions,
        }

    exist_filter = ExistsFilter(output_dir)
    dataset = (
        wds.WebDataset(
            data_path,
            handler=wds.warn_and_continue,
            nodesplitter=None,
            shardshuffle=False,
            empty_check=False,
        )
        .decode("pil", handler=wds.warn_and_continue)
        .map(partial(preprocess_fn, processor=processor), handler=wds.warn_and_continue)
    )
    if len(exist_filter.current_training_img_hashes) > 0:
        dataset = dataset.select(exist_filter)
    dataset = dataset.batched(
        batch_size,
        partial=False,
        collation_fn=collate_fn,
    )
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    def save_results(output_queue: queue.Queue, output_dir: pathlib.Path, processor):
        while True:
            try:
                item = output_queue.get(timeout=5)
                if item is None:
                    break
                original_captions, predictions, images, img_hashes = item
                predicted_captions = processor.batch_decode(
                    predictions,
                    skip_special_tokens=False,
                )
                for caption, pred_caption, image, img_hash in zip(
                    original_captions, predicted_captions, images, img_hashes
                ):
                    processed_caption = processor.post_process_generation(
                        pred_caption, task=prompt, image_size=(image.width, image.height)
                    )[prompt]
                    img_path = output_dir.joinpath(f"{img_hash}.jpg")
                    image.save(img_path)

                    caption_dict = {"original": caption, "predicted": processed_caption}
                    with output_dir.joinpath(f"{img_hash}_caption.json").open("w") as f:
                        json.dump(caption_dict, f, indent=4)

            except queue.Empty:
                continue

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_workers)
    save_future = save_thread.submit(save_results, output_queue, output_dir, processor)

    try:
        for _, batch_raw in tqdm(
            enumerate(dataloader),
            disable=not distributed_state.is_main_process,
        ):
            with distributed_state.split_between_processes(batch_raw) as batch:
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(distributed_state.device),
                    pixel_values=batch["pixel_values"].to(distributed_state.device, model.dtype),
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                output_queue.put(
                    (
                        batch["original_captions"],
                        outputs,
                        batch["images"],
                        batch["img_hashes"],
                    )
                )
    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()


if __name__ == "__main__":
    fire.Fire(main)
