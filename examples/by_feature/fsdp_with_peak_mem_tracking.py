# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import gc
import os
import threading

import evaluate
import psutil
import torch
from datasets import load_dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from accelerate import Accelerator, DistributedType, FullyShardedDataParallelPlugin
from accelerate.utils import is_npu_available, is_xpu_available


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#   - FSDP
#
# This example also demonstrates the checkpointing and sharding capabilities
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


# New Code #
# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# New Code #
# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.cuda.memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.xpu.memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            torch.npu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.npu.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            self.end = torch.xpu.memory_allocated()
            self.peak = torch.xpu.max_memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            self.end = torch.npu.memory_allocated()
            self.peak = torch.npu.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


# For testing only
if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
    from accelerate.test_utils.training import mocked_dataloaders

    get_dataloaders = mocked_dataloaders  # noqa: F811


def training_function(config, args):
    # For testing only
    if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
        config["num_epochs"] = 2

    # New Code #
    # Pass the advanced FSDP settings not part of the accelerate config by creating fsdp_plugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )

    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            project_dir=args.logging_dir,
            fsdp_plugin=fsdp_plugin,
        )
    else:
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    accelerator.print(accelerator.distributed_type)

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("fsdp_glue_no_trainer", experiment_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    datasets = load_dataset("glue", "mrpc")
    metric = evaluate.load("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, return_dict=True, low_cpu_mem_usage=True
    )

    # New Code #
    # For FSDP feature, it is highly recommended and efficient to prepare the model before creating optimizer
    model = accelerator.prepare(model)
    accelerator.print(model)

    # Instantiate optimizer
    # New Code #
    # For FSDP feature, at present it doesn't support multiple parameter groups,
    # so we need to create a single parameter group for the whole model
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=2e-4)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # New Code #
    # For FSDP feature, prepare everything except the model as we have already prepared the model
    # before creating the optimizer
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    overall_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            num_epochs -= int(training_difference.replace("epoch_", ""))
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            num_epochs -= resume_step // len(train_dataloader)
            # If resuming by step, we also need to know exactly how far into the DataLoader we went
            resume_step = (num_epochs * len(train_dataloader)) - resume_step

    # Now we train the model
    for epoch in range(num_epochs):
        # New Code #
        # context manager to track the peak memory usage during the training epoch
        with TorchTracemalloc() as tracemalloc:
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == 0:
                    if resume_step is not None and step < resume_step:
                        pass
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # accelerator.print(lr_scheduler.get_lr())

                overall_step += 1

                if isinstance(checkpointing_steps, int):
                    output_dir = f"step_{overall_step}"
                    if overall_step % checkpointing_steps == 0:
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
        # New Code #
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )
        # Logging the peak memory usage of the GPU to the tracker
        if args.with_tracking:
            accelerator.log(
                {
                    "train_total_peak_memory": tracemalloc.peaked + b2mb(tracemalloc.begin),
                },
                step=epoch,
            )

        # New Code #
        # context manager to track the peak memory usage during the evaluation
        with TorchTracemalloc() as tracemalloc:
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}:", eval_metric)
            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": eval_metric["accuracy"],
                        "f1": eval_metric["f1"],
                        "train_loss": total_loss.item() / len(train_dataloader),
                    },
                    step=epoch,
                )

            if checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
        # New Code #
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "Total Peak Memory consumed during the eval (max): {}".format(tracemalloc.peaked + b2mb(tracemalloc.begin))
        )
        # Logging the peak memory usage of the GPU to the tracker
        if args.with_tracking:
            accelerator.log(
                {
                    "eval_total_peak_memory": tracemalloc.peaked + b2mb(tracemalloc.begin),
                },
                step=epoch,
            )

    if args.with_tracking:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()
