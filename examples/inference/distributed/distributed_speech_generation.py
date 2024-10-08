
import os
import time
from typing import List, Dict
from pathlib import Path

import fire
import torch
import scipy.io.wavfile
import logging
from tqdm.auto import tqdm

from transformers import VitsModel, AutoTokenizer
from accelerate import PartialState
from accelerate.utils import gather_object, set_seed
from datasets import load_dataset


"""
put together together by damerajee.
Documentation: https://huggingface.co/docs/diffusers/main/en/training/distributed_inference

Run:

accelerate launch distributed_speech_generation.py --batch_size 8

"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def get_batches(items: List, batch_size: int, world_size: int) -> List:
    """
    Create batches that are evenly divisible by world_size to prevent hanging
    """
    effective_batch_size = batch_size * world_size
    num_items = len(items)
    num_complete_batches = num_items // effective_batch_size
    
    items = items[:num_complete_batches * effective_batch_size]
    
    return [
        items[i:i + effective_batch_size] 
        for i in range(0, len(items), effective_batch_size)
    ]

def save_audio(waveform: torch.Tensor, filepath: str, sampling_rate: int):
    """
    Save a waveform as a WAV file
    """
    try:
        # Convert to numpy and ensure proper scaling
        audio_data = waveform.float().numpy()
        # Normalize audio to prevent clipping
        audio_data = audio_data / max(abs(audio_data.min()), abs(audio_data.max()))
        # Save as WAV file
        scipy.io.wavfile.write(filepath, sampling_rate, audio_data)
    except Exception as e:
        logger.error(f"Error saving audio file {filepath}: {str(e)}")
        raise

def main(
    save_dir: str = "./generated_speech",
    model_id :str = "facebook/mms-tts-eng",
    seed: int = 1,
    batch_size: int = 4,
    dataset_name: str = "svjack/pokemon-blip-captions-en-zh",
    dataset_split: str = "train",
    max_samples: int = 100,
    dtype: str = "fp16"

):


    
    # Initialize distributed state
    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    process_idx = distributed_state.process_index
    
    logger.info(f"Process {process_idx}/{world_size} initialized on {distributed_state.device}")
    
    # Set seeds for reproducibility
    set_seed(seed)
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id, torch_dtype=DTYPE_MAP[dtype]).to(distributed_state.device)
        
        save_dir = Path(save_dir) / f"eng_{START_TIME}"

        # Load dataset with text samples
        logger.info(f"Process {process_idx}: Loading dataset {dataset_name}")
        ds = load_dataset(dataset_name, split=dataset_split)
        
        # Get text samples from dataset
        text_samples = ds["en_text"][:max_samples]
        
        # Create properly sized batches
        data_loader = get_batches(text_samples, batch_size, world_size)
        total_batches = len(data_loader)
        
        logger.info(f"Process {process_idx}: Created {total_batches} batches")

        # Create output directory on main process
        if distributed_state.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {save_dir}")

        # Synchronize processes
        distributed_state.wait_for_everyone()
        
        # Initialize progress bar only on main process
        pbar = tqdm(
            total=total_batches,
            disable=not distributed_state.is_main_process,
            desc=f"Generating eng speech"
        )

        count = 0
        for batch_idx, texts_raw in enumerate(data_loader):
            generated_audio = []
            
            try:
                with distributed_state.split_between_processes(texts_raw) as texts:
                    for text in texts:
                        # Tokenize and generate speech
                        inputs = tokenizer(text, return_tensors="pt").to(distributed_state.device)
                        with torch.no_grad():
                            output = model(**inputs).waveform
                        
                        generated_audio.append({
                            'text': text,
                            'waveform': output.cpu(),
                        })

                # Synchronize processes
                distributed_state.wait_for_everyone()

                # Gather results from all processes
                all_generated = gather_object(generated_audio)

                # Save results on main process
                if distributed_state.is_main_process:
                    for audio_data in all_generated:
                        count += 1
                        
                        # Create audio file path
                        audio_path = save_dir / f"audio_{count:04d}.wav"
                        text_path = save_dir / f"text_{count:04d}.txt"
                        
                        # Save audio file
                        save_audio(
                            audio_data['waveform'][0],  # Remove batch dimension
                            str(audio_path),
                            model.config.sampling_rate
                        )
                        
                        # Save corresponding text
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(audio_data['text'])
                    
                    pbar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

        # Clean up
        distributed_state.wait_for_everyone()
        if distributed_state.is_main_process:
            pbar.close()
            logger.info(f"Speech Generation Finished. Saved {count} files in {save_dir}")

    except Exception as e:
        logger.error(f"Critical error in process {process_idx}: {str(e)}")
        raise


if __name__ == "__main__":
    fire.Fire(main)