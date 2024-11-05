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
import time
from concurrent.futures import ThreadPoolExecutor

import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

from accelerate import PartialState


START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


"""
Example:

accelerate launch llava_next_video.py
"""


def save_results(output_queue: queue.Queue, output_dir: pathlib.Path):
    count = 0
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break
            example_file = f"example_{count}"
            temp_dir = os.path.join(output_dir, example_file)

            metadata = {
                "caption": item[0],
                "generated_answer": item[1],
            }
            with open(temp_dir, "w") as f:
                json.dump(metadata, f, indent=4)
            count += 1

        except queue.Empty:
            continue


def get_batches(captions, batch_size):
    num_batches = (len(captions) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(captions))
        batch = captions[start_index:end_index]
        batches.append(batch)

    return batches


def main(
    model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
    save_dir: str = "./evaluation/examples",
    max_captions: int = 10,
    max_new_tokens: int = 100,
    batch_size: int = 4,
    dtype: str = "fp16",
    num_workers: int = 1,
    low_mem: bool = True,
):
    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()

    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=DTYPE_MAP[dtype], low_cpu_mem_usage=low_mem, device_map=distributed_state.device
    )

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    captions = load_dataset("nkp37/OpenVid-1M", split="train")["caption"]
    reduced_captions = captions[: min(len(captions), max_captions)]
    batches = get_batches(reduced_captions, batch_size)

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_workers)
    save_future = save_thread.submit(save_results, output_queue, save_dir)
    for _, caption_batch in tqdm(enumerate(batches), total=len(batches)):
        try:
            with distributed_state.split_between_processes(caption_batch) as caption:
                input = processor(caption, padding=True, return_tensors="pt").to(model.device)
                output = model.generate(**input, max_new_tokens=max_new_tokens)
                generated_text = processor.batch_decode(output, skip_special_tokens=True)
                output_queue.put((caption, generated_text))
        finally:
            output_queue.put(None)
            save_thread.shutdown(wait=True)

    save_future.result()


if __name__ == "__main__":
    fire.Fire(main)
