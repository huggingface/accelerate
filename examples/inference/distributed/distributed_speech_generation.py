import os
import time
from typing import List
from pathlib import Path

import fire
import torch
import scipy.io.wavfile
import logging
from tqdm import tqdm

from transformers import VitsModel, AutoTokenizer
from accelerate import PartialState
from accelerate.utils import gather_object, set_seed
from datasets import load_dataset

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
    batches = [
        items[i:i + effective_batch_size] 
        for i in range(0, len(items), effective_batch_size)
    ]
    return batches

def save_audio(waveform: torch.Tensor, filepath: str, sampling_rate: int):
    """
    Save a waveform as a WAV file
    """
    try:
        audio_data = waveform.float().numpy()
        audio_data = audio_data / max(abs(audio_data.min()), abs(audio_data.max()))
        scipy.io.wavfile.write(filepath, sampling_rate, audio_data)
    except Exception as e:
        logger.error(f"Error saving audio file {filepath}: {str(e)}")
        raise

def main(
    model_id: str = "facebook/mms-tts-eng",
    save_dir: str = "./generated_speech",
    seed: int = 1,
    batch_size: int = 4,
    dataset_name: str = "svjack/pokemon-blip-captions-en-zh",
    dataset_split: str = "train",
    max_samples: int = 100,
    dtype: str = "fp16"
):
    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    process_idx = distributed_state.process_index
    
    logger.info(f"Process {process_idx}/{world_size} initialized on {distributed_state.device}")
    
    set_seed(seed)
    
    try:
        logger.info(f"Process {process_idx}: Loading tokenizer and model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(
            model_id,
            device_map=distributed_state.device,
            torch_dtype=DTYPE_MAP[dtype]
        )
        
        save_dir = Path(save_dir) / f"eng_{START_TIME}"

        logger.info(f"Process {process_idx}: Loading dataset {dataset_name}")
        ds = load_dataset(dataset_name, split=dataset_split)
        
        text_samples = ds["en_text"][:max_samples]
        
        data_loader = get_batches(text_samples, batch_size, world_size)
        total_batches = len(data_loader)
        
        logger.info(f"Process {process_idx}: Created {total_batches} batches")

        if distributed_state.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {save_dir}")

        logger.info(f"Process {process_idx}: Waiting for all processes")

        distributed_state.wait_for_everyone()  
        logger.info(f"Process {process_idx}: All processes ready")

        pbar = tqdm(
            total=total_batches,
            disable=not distributed_state.is_main_process,
            desc="Generating eng speech"
        )

        count = 0
        for batch_idx, texts_raw in enumerate(data_loader):
            generated_audio = []
            
            try:
                logger.info(f"Process {process_idx}: Processing batch {batch_idx}")
                with distributed_state.split_between_processes(texts_raw) as texts:
                    for text in texts:
                        inputs = tokenizer(text, return_tensors="pt").to(distributed_state.device)
                        try:
                            with torch.no_grad():
                                output = model(**inputs).waveform
                            
                            generated_audio.append({
                                'text': text,
                                'waveform': output.cpu(),
                            })
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logger.warning(f"CUDA out of memory in process {process_idx}. Skipping this sample.")
                                torch.cuda.empty_cache()
                            else:
                                raise

                logger.info(f"Process {process_idx}: Waiting for all processes after batch {batch_idx}")
                distributed_state.wait_for_everyone()  
                logger.info(f"Process {process_idx}: All processes finished batch {batch_idx}")

                all_generated = gather_object(generated_audio)

                if distributed_state.is_main_process:
                    for audio_data in all_generated:
                        count += 1
                        temp_dir = save_dir / f"example_{count}"
                        temp_dir.mkdir(parents=True, exist_ok=True)

                        audio_path = temp_dir / f"audio_{count}.wav"
                        text_path = temp_dir / f"text_{count}.txt"

                        save_audio(
                            audio_data['waveform'][0],  
                            str(audio_path),
                            model.config.sampling_rate
                        )

                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(audio_data['text'])

                    pbar.update(1)
                    logger.info(f"Main process: Completed batch {batch_idx}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} in process {process_idx}: {str(e)}")
                continue

        logger.info(f"Process {process_idx}: Finished all batches. Waiting for other processes.")
        distributed_state.wait_for_everyone()  
        if distributed_state.is_main_process:
            pbar.close()
            logger.info(f"Speech Generation Finished. Saved {count} files in {save_dir}")

    except Exception as e:
        logger.error(f"Critical error in process {process_idx}: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(main)