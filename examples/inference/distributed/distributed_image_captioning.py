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
Additional requirements: einops 
pip install  einops 

put together together by damerajee.
Documentation: https://huggingface.co/docs/diffusers/main/en/training/distributed_inference

Run:

accelerate launch distributed_image_generation.py --batch_size 8

"""

import os
import time

import fire
import torch
from datasets import load_dataset
from transformers import BlipForConditionalGeneration, BlipProcessor
from tqdm import tqdm
import logging 
from typing import List 

from accelerate import PartialState
from accelerate.utils import gather_object, set_seed


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def get_batches(items: List, batch_size: int, world_size: int) -> List:
    """
    Create batches that are evenly divisible by world_size to prevent hanging
    """
    # Adjust batch_size to be divisible by world_size
    effective_batch_size = batch_size * world_size
    # Calculate number of complete batches
    num_items = len(items)
    num_complete_batches = num_items // effective_batch_size
    
    # Trim the items to be evenly divisible
    items = items[:num_complete_batches * effective_batch_size]
    
    # Create batches
    batches = [
        items[i:i + effective_batch_size] 
        for i in range(0, len(items), effective_batch_size)
    ]
    
    return batches

def main(
    model_id: str = "Salesforce/blip-image-captioning-base",
    save_dir: str = "./evaluation/captions",
    seed: int = 1,
    batch_size: int = 4,
    dtype: str = "fp16",
    dataset_name: str = "uoft-cs/cifar10",
    dataset_split: str = "train"
):
    # Initialize distributed state
    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    process_idx = distributed_state.process_index
    
    logger.info(f"Process {process_idx}/{world_size} initialized on {distributed_state.device}")
    
    # Set seeds for reproducibility
    set_seed(seed)
    
    try:
        # Load processor and model for captioning
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            device_map=distributed_state.device,
            torch_dtype=DTYPE_MAP[dtype]
        )
        
        save_dir = save_dir + f"_{START_TIME}"

        # Load dataset
        logger.info(f"Process {process_idx}: Loading dataset {dataset_name}")
        ds = load_dataset(dataset_name, split=dataset_split)
        ds = ds.select(range(1000))    
        # Create properly sized batches
        data_loader = get_batches(ds["img"], batch_size, world_size)
    
        total_batches = len(data_loader)
        
        logger.info(f"Process {process_idx}: Created {total_batches} batches")

        # Create output directory on main process
        if distributed_state.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Created output directory: {save_dir}")

        # Synchronize processes
        distributed_state.wait_for_everyone()
        
        # Initialize progress bar only on main process
        pbar = tqdm(
            total=total_batches,
            disable=not distributed_state.is_main_process,
            desc="Processing batches"
        )

        count = 0
        for batch_idx, images_raw in enumerate(data_loader):
            input_images = []
            
            try:
                with distributed_state.split_between_processes(images_raw) as images:
                    # Process images and generate captions
                    inputs = processor(images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(distributed_state.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(**inputs)
                    captions = processor.batch_decode(outputs, skip_special_tokens=True)
                    input_images.extend(images)

                # Synchronize processes
                distributed_state.wait_for_everyone()

                # Gather results from all processes
                captions = gather_object(captions)
                input_images = gather_object(input_images)

                # Save results on main process
                if distributed_state.is_main_process:
                    for caption, img in zip(captions, input_images):
                        count += 1
                        temp_dir = os.path.join(save_dir, f"example_{count}")
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        img.save(os.path.join(temp_dir, f"image_{count}.png"))
                        with open(os.path.join(temp_dir, "caption.txt"), "w") as f:
                            f.write(caption)
                    
                    # Update progress bar
                    pbar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

        # Clean up
        distributed_state.wait_for_everyone()
        if distributed_state.is_main_process:
            pbar.close()
            logger.info(f"Caption Generation Finished. Saved in {save_dir}")
            logger.info(f"Processed {count} images in total")

    except Exception as e:
        logger.error(f"Critical error in process {process_idx}: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(main)