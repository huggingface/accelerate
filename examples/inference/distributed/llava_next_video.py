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

import fire
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import numpy as np
import torch
import time
from accelerate import PartialState
import os
import av
from huggingface_hub import hf_hub_download
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import pathlib

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


"""
Example:

accelerate launch llava_next_video.py 
"""


def main(
    model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
    save_dir: str = "./evaluation/examples",
    dtype: str = "fp16",
    num_workers: int = 1,
    low_mem: bool = True,
):
    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()

    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype[DTYPE_MAP], low_cpu_mem_usage=low_mem, device_map=distributed_state.device
    )

    if distributed_state.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created successfully.")
        else:
            print(f"Directory '{save_dir}' already exists.")

    # Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
    video_path = hf_hub_download(
        repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
    )
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Why is this video funny?"},
                    {"type": "video"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Why is this video sad?"},
                    {"type": "video"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you think about this video?"},
                    {"type": "video"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Do you like this video?"},
                    {"type": "video"},
                ],
            }
        ],
    ]

    formatted_prompts = [
        processor.apply_chat_template(conversations[i], add_generation_prompt=True)
        for i in range(0, len(conversations))
    ]

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
                    "prompt": item[0],
                    "generated_answer": item[1],
                }
                with open(temp_dir, "w") as f:
                    json.dump(metadata, f, indent=4)
                count += 1

            except queue.Empty:
                continue

    distributed_state.num_processes = len(formatted_prompts)
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_workers)
    save_future = save_thread.submit(save_results, output_queue, save_dir)

    try:
        with distributed_state.split_between_processes(formatted_prompts) as prompt:
            input = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)
            output = model.generate(**input, max_new_tokens=60)
            generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
            output_queue.put((prompt, generated_text))
    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


if __name__ == "__main__":
    fire.Fire(main)
