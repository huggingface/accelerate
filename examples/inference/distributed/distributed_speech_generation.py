import json
import os
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Union

import fire
import torch
import scipy.io.wavfile
from tqdm import tqdm
from transformers import VitsModel, AutoTokenizer
from datasets import load_dataset
from accelerate import PartialState

"""
Requirements: transformers accelerate fire  scipy datasets
pip install transformers accelerate fire scipy datasets 
Example usage:

accelerate launch --num_processes=2 pokemon_speech.py --output_path outputs --batch_size 8 --num_workers 2 --dataset_split train
"""


def main(
    output_path: str="speech_data",
    batch_size: int=8,
    num_workers:int=2,
    dataset_split: str = "train",
    model_name: str = "facebook/mms-tts-eng",
    max_text_length: int = 200,
):
    output_dir = pathlib.Path(output_path)
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        output_dir.mkdir(exist_ok=True)

    # Load model and tokenizer
    model = VitsModel.from_pretrained(
        model_name,
        device_map=distributed_state.device,
        torch_dtype=torch.float32,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ExistsFilter:
        def __init__(self, output_dir: Union[pathlib.Path, str]):
            current_files = [f.split(".wav")[0] for f in os.listdir(output_dir) if f.endswith(".wav")]
            self.processed_files = set(current_files)
            if distributed_state.is_main_process:
                print(f"Existing audio files found: {len(self.processed_files)}.")

        def __call__(self, x):
            return x["id"] not in self.processed_files

    def load_pokemon_data(split: str):
        """Load Pokemon descriptions from the dataset"""
        ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split=split)
        
        # Create dataset of dictionaries
        dataset = []
        for idx, text in enumerate(ds["en_text"]):
            if len(text.strip()) > 0:  # Skip empty descriptions
                dataset.append({
                    "id": f"pokemon_{idx:06d}",
                    "text": text.strip()[:max_text_length],  # Truncate long descriptions
                    "original_text": text.strip()  # Keep original for metadata
                })
        return dataset

    def preprocess_fn(sample, tokenizer):
        inputs = tokenizer(
            sample["text"],
            padding=False,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "id": sample["id"],
            "text": sample["text"],
            "original_text": sample["original_text"]
        }

    def collate_fn(examples):
        """Collate batch of examples with proper padding"""
        # Find max length in this batch
        max_length = max(len(example["input_ids"]) for example in examples)

        # Pad sequences to max_length
        input_ids_list = []
        attention_mask_list = []

        for example in examples:
            # Get current lengths
            curr_len = len(example["input_ids"])
            padding_length = max_length - curr_len

            # Pad sequences
            padded_input_ids = example["input_ids"] + [tokenizer.pad_token_id] * padding_length
            padded_attention_mask = example["attention_mask"] + [0] * padding_length

            input_ids_list.append(padded_input_ids)
            attention_mask_list.append(padded_attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(input_ids_list,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list,dtype=torch.long)

        ids = [example["id"] for example in examples]
        texts = [example["text"] for example in examples]
        original_texts = [example["original_text"] for example in examples]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ids": ids,
            "texts": texts,
            "original_texts": original_texts
        }
    def create_dataloader(dataset, batch_size):
        """Create dataloader with preprocessing"""
        processed_dataset = [preprocess_fn(item, tokenizer) for item in dataset]

        # Split dataset for distributed processing
        if distributed_state.num_processes > 1:
            chunk_size = len(processed_dataset) // distributed_state.num_processes
            start_idx = distributed_state.process_index * chunk_size
            end_idx = start_idx + chunk_size if distributed_state.process_index < distributed_state.num_processes - 1 else len(processed_dataset)
            processed_dataset = processed_dataset[start_idx:end_idx]

        # Create batches
        batches = []
        for i in range(0, len(processed_dataset), batch_size):
            batch = processed_dataset[i:i + batch_size]
            batches.append(collate_fn(batch))
        return batches


    def save_results(output_queue: queue.Queue, output_dir: pathlib.Path, sampling_rate: int):
        while True:
            try:
                item = output_queue.get(timeout=5)
                if item is None:
                    break
                waveforms, ids, texts, original_texts = item
                
                # Save each audio file and its metadata
                for waveform, file_id, text, original_text in zip(waveforms, ids, texts, original_texts):
                    # Save audio
                    wav_path = output_dir / f"{file_id}.wav"
                    scipy.io.wavfile.write(
                        wav_path,
                        rate=sampling_rate,
                        data=waveform.cpu().float().numpy()
                    )
                    
                    # Save metadata with both truncated and original text
                    metadata = {
                        "text_used": text,
                        "original_text": original_text,
                        "model": model_name,
                        "sampling_rate": sampling_rate
                    }
                    metadata_path = output_dir / f"{file_id}_metadata.json"
                    with metadata_path.open("w") as f:
                        json.dump(metadata, f, indent=4)
                        
            except queue.Empty:
                continue

    # Load and filter data
    dataset = load_pokemon_data(dataset_split)
    exist_filter = ExistsFilter(output_dir)
    dataset = [item for item in dataset if exist_filter(item)]
    
    if distributed_state.is_main_process:
        print(f"Processing {len(dataset)} Pokemon descriptions")
    
    # Create dataloader
    batches = create_dataloader(dataset, batch_size)

    # Setup output queue and save thread
    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_workers)
    save_future = save_thread.submit(
        save_results,
        output_queue,
        output_dir,
        model.config.sampling_rate
    )

    try:
        for batch in tqdm(
            batches,
            disable=not distributed_state.is_main_process,
            desc="Generating Pokemon descriptions"
        ):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(distributed_state.device, dtype=torch.long),
                    attention_mask=batch["attention_mask"].to(distributed_state.device, dtype=torch.long)
                        ).waveform

                output_queue.put((
                    outputs,
                    batch["ids"],
                    batch["texts"],
                    batch["original_texts"]
                ))
    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()

if __name__ == "__main__":
    fire.Fire(main)


"""
To run the speech generation 

import scipy.io.wavfile
import numpy as np
from IPython.display import Audio

sample_rate, audio_data = scipy.io.wavfile.read('path_to_you_wav_file.wav')

audio_data = audio_data.astype(np.float32) / 32762.0
Audio(audio_data, rate=sample_rate)
"""