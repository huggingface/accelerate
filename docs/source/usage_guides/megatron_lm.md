<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->


# Megatron-LM

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) enables training large transformer language models at scale.
It provides efficient tensor, pipeline and sequence based model parallelism for pre-training transformer based
Language Models such as [GPT](https://arxiv.org/abs/2005.14165) (Decoder Only), [BERT](https://arxiv.org/pdf/1810.04805.pdf) (Encoder Only) and [T5](https://arxiv.org/abs/1910.10683) (Encoder-Decoder).
For detailed information and how things work behind the scene please refer the github [repo](https://github.com/NVIDIA/Megatron-LM).

## What is integrated?

Accelerate integrates following feature of Megatron-LM to enable large scale pre-training/finetuning
of BERT (Encoder), GPT (Decoder) or T5 models (Encoder and Decoder):

a. **Tensor Parallelism (TP)**: Reduces memory footprint without much additional communication on intra-node ranks.
Each tensor is split into multiple chunks with each shard residing on separate GPU. At each step, the same mini-batch of data is processed
independently and in parallel by each shard followed by syncing across all GPUs (`all-reduce` operation). 
In a simple transformer layer, this leads to 2 `all-reduces` in the forward path and 2 in the backward path.
For more details, please refer research paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf) and 
this section of 🤗 blogpost [The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#tensor-parallelism).


b. **Pipeline Parallelism (PP)**: Reduces memory footprint and enables large scale training via inter-node parallelization. 
Reduces the bubble of naive PP via PipeDream-Flush schedule/1F1B schedule and Interleaved 1F1B schedule. 
Layers are distributed uniformly across PP stages. For example, if a model has `24` layers and we have `4` GPUs for
pipeline parallelism, each GPU will have `6` layers (24/4). For more details on schedules to reduce the idle time of PP,
please refer to the research paper [Efficient Large-Scale Language Model Training on GPU Clusters
Using Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf) and 
this section of 🤗 blogpost [The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism).

c. **Sequence Parallelism (SP)**: Reduces memory footprint without any additional communication. Only applicable when using TP.
It reduces activation memory required as it prevents the same copies to be on the tensor parallel ranks 
post `all-reduce` by replacing then with `reduce-scatter` and `no-op` operation would be replaced by `all-gather`. 
As `all-reduce = reduce-scatter + all-gather`, this saves a ton of activation memory at no added communication cost. 
To put it simply, it shards the outputs of each transformer layer along sequence dimension, e.g., 
if the sequence length is `1024` and the TP size is `4`, each GPU will have `256` tokens (1024/4) for each sample. 
This increases the batch size that can be supported for training. For more details, please refer to the research paper
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf). 

d. **Data Parallelism (DP)** via Distributed Optimizer: Reduces the memory footprint by sharding optimizer states and gradients across DP ranks
(versus the traditional method of replicating the optimizer state across data parallel ranks). 
For example, when using Adam optimizer with mixed-precision training, each parameter accounts for 12 bytes of memory.
This gets distributed equally across the GPUs, i.e., each parameter would account for 3 bytes (12/4) if we have 4 GPUs.
For more details, please refer the research paper [ZeRO: Memory Optimizations Toward Training Trillion
Parameter Models](https://arxiv.org/pdf/1910.02054.pdf) and following section of 🤗 blog 
[The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#zero-data-parallelism).

e. **Selective Activation Recomputation**: Reduces the memory footprint of activations significantly via smart activation checkpointing.
It doesn't store activations occupying large memory while being fast to recompute thereby achieving great tradeoff between memory and recomputation.
For example, for GPT-3, this leads to 70% reduction in required memory for activations at the expense of
only 2.7% FLOPs overhead for recomputation of activations. For more details, please refer to the research paper 
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf).

f. **Fused Kernels**: Fused Softmax, Mixed Precision Fused Layer Norm and  Fused gradient accumulation to weight gradient computation of linear layer.
PyTorch JIT compiled Fused GeLU and Fused Bias+Dropout+Residual addition.

g. **Support for Indexed datasets**: Efficient binary format of datasets for large scale training. Support for the `mmap`, `cached` index file and the `lazy` loader format.

h. **Checkpoint reshaping and interoperability**: Utility for reshaping Megatron-LM checkpoints of variable 
tensor and pipeline parallel sizes to the beloved 🤗 Transformers sharded checkpoints as it has great support with plethora of tools
such as 🤗 Accelerate Big Model Inference, Megatron-DeepSpeed Inference etc. 
Support is also available for converting 🤗 Transformers sharded checkpoints to Megatron-LM checkpoint of variable tensor and pipeline parallel sizes
for large scale training.  


## Pre-Requisites 

You will need to install the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases and the nltk library.
See [documentation](https://github.com/NVIDIA/Megatron-LM#setup) for more details. 
Another way to setup the environment is to pull an NVIDIA PyTorch Container that comes with all the required installations from NGC.

Below is a step-by-step method to set up the conda environment:

1. Create a virtual environment
```
conda create --name ml
```

2. Assuming that the machine has CUDA 11.3 installed, installing the corresponding PyTorch GPU Version
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. Install Nvidia APEX
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

4. Installing Megatron-LM

```
pip install git+https://github.com/huggingface/Megatron-LM.git
```

## Accelerate Megatron-LM Plugin

Important features are directly supported via the `accelerate config` command. 
An example of thr corresponding questions for using Megatron-LM features is shown below:

```bash
:~$ accelerate config --config_file "megatron_gpt_config.yaml"
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
Do you want to use DeepSpeed? [yes/NO]: 
Do you want to use FullyShardedDataParallel? [yes/NO]: 
Do you want to use Megatron-LM ? [yes/NO]: yes
What is the Tensor Parallelism degree/size? [1]:2
Do you want to enable Sequence Parallelism? [YES/no]: 
What is the Pipeline Parallelism degree/size? [1]:2
What is the number of micro-batches? [1]:2
Do you want to enable selective activation recomputation? [YES/no]: 
Do you want to use distributed optimizer which shards optimizer state and gradients across data pralellel ranks? [YES/no]: 
What is the gradient clipping value based on global L2 Norm (0 to disable)? [1.0]: 
How many GPU(s) should be used for distributed training? [1]:4
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: bf16
```

The resulting config is shown below:

```
~$ cat megatron_gpt_config.yaml 
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MEGATRON_LM
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config:
  megatron_lm_gradient_clipping: 1.0
  megatron_lm_num_micro_batches: 2
  megatron_lm_pp_degree: 2
  megatron_lm_recompute_activations: true
  megatron_lm_sequence_parallelism: true
  megatron_lm_tp_degree: 2
  megatron_lm_use_distributed_optimizer: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```

We will take the example of GPT pre-training. The minimal changes required to the official `run_clm_no_trainer.py` 
to use Megatron-LM are as follows:

1. As Megatron-LM uses its own implementation of Optimizer, the corresponding scheduler compatible with it needs to be used.
As such, support for only the Megatron-LM's scheduler is present. User will need to create `accelerate.utils.MegatronLMDummyScheduler`.
Example is given below:

```python
from accelerate.utils import MegatronLMDummyScheduler

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    lr_scheduler = MegatronLMDummyScheduler(
        optimizer=optimizer,
        total_num_steps=args.max_train_steps,
        warmup_num_steps=args.num_warmup_steps,
    )
else:
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
```

2. Getting the details of the total batch size now needs to be cognization of tensor and pipeline parallel sizes.
Example of getting the effective total batch size is shown below:

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    total_batch_size = accelerator.state.megatron_lm_plugin.global_batch_size
else:
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
```

3. When using Megatron-LM, the losses are already averaged across the data parallel group

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    losses.append(loss)
else:
    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    losses = torch.tensor(losses)
else:
    losses = torch.cat(losses)
```

4. For Megatron-LM, we need to save the model using `accelerator.save_state`

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    accelerator.save_state(args.output_dir)
else:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
```

That's it! We are good to go 🚀. Please find the example script in the examples folder at the path `accelerate/examples/by_feature/megatron_lm_gpt_pretraining.py`.
Let's run it for `gpt-large` model architecture using 4 A100-80GB GPUs.

```bash
accelerate launch --config_file megatron_gpt_config.yaml \
examples/by_feature/megatron_lm_gpt_pretraining.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--block_size 1024 \
--learning_rate 5e-5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 5 \
--with_tracking \
--report_to "wandb" \
--output_dir "awesome_model"
```

Below are some important excerpts from the output logs:

```bash
Loading extension module fused_dense_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 3.569 seconds
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
Building gpt model in the pre-training mode.
The Megatron LM model weights are initialized at random in `accelerator.prepare`. Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup.
Preparing dataloader
Preparing dataloader
Preparing model
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 210753280
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 209445120
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 210753280
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 209445120
Preparing optimizer
Preparing scheduler
> learning rate decay style: linear
10/10/2022 22:57:22 - INFO - __main__ - ***** Running training *****
10/10/2022 22:57:22 - INFO - __main__ -   Num examples = 2318
10/10/2022 22:57:22 - INFO - __main__ -   Num Epochs = 5
10/10/2022 22:57:22 - INFO - __main__ -   Instantaneous batch size per device = 24
10/10/2022 22:57:22 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 48
10/10/2022 22:57:22 - INFO - __main__ -   Gradient Accumulation steps = 1
10/10/2022 22:57:22 - INFO - __main__ -   Total optimization steps = 245
 20%|████████████▍                                                 | 49/245 [01:04<04:09,  1.27s/it]
 10/10/2022 22:58:29 - INFO - __main__ - epoch 0: perplexity: 1222.1594275215962 eval_loss: 7.10837459564209
 40%|████████████████████████▊                                     | 98/245 [02:10<03:07,  1.28s/it]
 10/10/2022 22:59:35 - INFO - __main__ - epoch 1: perplexity: 894.5236583794557 eval_loss: 6.796291351318359
 60%|████████████████████████████████████▌                        | 147/245 [03:16<02:05,  1.28s/it]
 10/10/2022 23:00:40 - INFO - __main__ - epoch 2: perplexity: 702.8458788508042 eval_loss: 6.555137634277344
 80%|████████████████████████████████████████████████▊            | 196/245 [04:22<01:02,  1.28s/it]
 10/10/2022 23:01:46 - INFO - __main__ - epoch 3: perplexity: 600.3220028695281 eval_loss: 6.39746618270874
100%|█████████████████████████████████████████████████████████████| 245/245 [05:27<00:00,  1.28s/it]
```

There are a large number of other options/features that one can set using `accelerate.utils.MegatronLMPlugin`.

## Advanced features to leverage writing custom train step and Megatron-LM Indexed Datasets

For leveraging more features, please go through below details.

1. Below is an example of changes required to customize the Train Step while using Megatron-LM. 
You will implement the `accelerate.utils.AbstractTrainStep` or inherit from their corresponding children 
`accelerate.utils.GPTTrainStep`, `accelerate.utils.BertTrainStep` or `accelerate.utils.T5TrainStep`.

```python
from accelerate.utils import MegatronLMDummyScheduler, GPTTrainStep, avg_losses_across_data_parallel_group


# Custom loss function for the Megatron model
class GPTTrainStepWithCustomLoss(GPTTrainStep):
    def __init__(self, megatron_args, **kwargs):
        super().__init__(megatron_args)
        self.kwargs = kwargs

    def get_loss_func(self):
        def loss_func(inputs, loss_mask, output_tensor):
            batch_size, seq_length = output_tensor.shape
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = losses.view(-1) * loss_mask

            # Resize and average loss per sample
            loss_per_sample = loss.view(batch_size, seq_length).sum(axis=1)
            loss_mask_per_sample = loss_mask.view(batch_size, seq_length).sum(axis=1)
            loss_per_sample = loss_per_sample / loss_mask_per_sample

            # Calculate and scale weighting
            weights = torch.stack([(inputs == kt).float() for kt in self.kwargs["keytoken_ids"]]).sum(axis=[0, 2])
            weights = 1.0 + self.kwargs["alpha"] * weights
            # Calculate weighted average
            weighted_loss = (loss_per_sample * weights).mean()

            # Reduce loss across data parallel groups
            averaged_loss = avg_losses_across_data_parallel_group([weighted_loss])

            return weighted_loss, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.loss_func, tokens, loss_mask)

        return forward_step


def main():
    # Custom loss function for the Megatron model
    keytoken_ids = []
    keywords = ["plt", "pd", "sk", "fit", "predict", " plt", " pd", " sk", " fit", " predict"]
    for keyword in keywords:
        ids = tokenizer([keyword]).input_ids[0]
        if len(ids) == 1:
            keytoken_ids.append(ids[0])
    accelerator.print(f"Keytoken ids: {keytoken_ids}")
    accelerator.state.megatron_lm_plugin.custom_train_step_class = GPTTrainStepWithCustomLoss
    accelerator.state.megatron_lm_plugin.custom_train_step_kwargs = {
        "keytoken_ids": keytoken_ids,
        "alpha": 0.25,
    }
```

2. For using the Megatron-LM datasets, a few more changes are required. Dataloaders for these datasets
are available only on rank 0 of each tensor parallel group. As such, there are rank where dataloader won't be
avaiable and this requires tweaks to the training loop. Being able to do all this shows how
felixble and extensible 🤗 Accelerate is. The changes required are as follows.

a. For Megatron-LM indexed datasets, we need to use `MegatronLMDummyDataLoader` 
and pass the required dataset args to it such as `data_path`, `seq_length` etc. 
See [here](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L804) for the list of available args. 
    
```python
from accelerate.utils import MegatronLMDummyDataLoader

megatron_dataloader_config = {
    "data_path": args.data_path,
    "splits_string": args.splits_string,
    "seq_length": args.block_size,
    "micro_batch_size": args.per_device_train_batch_size,
}
megatron_dataloader = MegatronLMDummyDataLoader(**megatron_dataloader_config)
accelerator.state.megatron_lm_plugin.megatron_dataset_flag = True
```

b. `megatron_dataloader` is repeated 3 times to get training, validation and test dataloaders
as per the `args.splits_string` proportions
    
```python
model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, _ = accelerator.prepare(
    model, optimizer, lr_scheduler, megatron_dataloader, megatron_dataloader, megatron_dataloader
)
```

c. Changes to training and evaluation loops as dataloader is only available on tensor parallel ranks 0
So, we need to iterate only if the dataloader isn't `None` else provide empty dict
As such, we loop using `while` loop and break when `completed_steps` is equal to `args.max_train_steps`
This is similar to the Megatron-LM setup wherein user has to provide `max_train_steps` when using Megaton-LM indexed datasets.
This displays how flexible and extensible 🤗 Accelerate is.

```python
while completed_steps < args.max_train_steps:
    model.train()
    batch = next(train_dataloader) if train_dataloader is not None else {}
    outputs = model(**batch)
    loss = outputs.loss
    ...

    if completed_steps % eval_interval == 0:
        eval_completed_steps = 0
        losses = []
        while eval_completed_steps < eval_iters:
            model.eval()
            with torch.no_grad():
                batch = next(eval_dataloader) if eval_dataloader is not None else {}
                outputs = model(**batch)
```

    
## Utility for Checkpoint reshaping and interoperability

1. The scripts for these are present in 🤗 Transformers library under respective models. 
Currently, it is available for GPT model [checkpoint_reshaping_and_interoperability.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py)

2. Below is an example of conversion of checkpoint from Megatron-LM to universal 🤗 Transformers sharded checkpoint.
```bash
python checkpoint_reshaping_and_interoperability.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "gpt/iter_0005000" \
--save_path "gpt/trfs_checkpoint" \
--max_shard_size "200MB" \
--tokenizer_name "gpt2" \
--print-checkpoint-structure
```

3. Conversion of checkpoint from transformers to megatron with `tp_size=2`, `pp_size=2` and `dp_size=2`.
```bash
python checkpoint_utils/megatgron_gpt2/checkpoint_reshaping_and_interoperability.py \
--load_path "gpt/trfs_checkpoint" \
--save_path "gpt/megatron_lm_checkpoint" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 2 \
--target_data_parallel_size 2 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 128 \
--use_distributed_optimizer \
--print-checkpoint-structure
```

## Megatron-LM GPT models support returning logits and `megatron_generate` function for text generation

1. Returning logits require setting `require_logits=True` in MegatronLMPlugin as shown below. 
These would be available on the in the last stage of pipeline.
```python
megatron_lm_plugin = MegatronLMPlugin(return_logits=True)
```

2. `megatron_generate` method for Megatron-LM GPT model: This will use Tensor and Pipeline Parallelism to complete 
generations for a batch of inputs when using greedy with/without top_k/top_p sampling and for individual prompt inputs when using beam search decoding. 
Only a subset of features of transformers generate is supported. This will help in using large models via tensor and pipeline parallelism 
for generation (already does key-value caching and uses fused kernels by default).
This requires data parallel size to be 1, sequence parallelism and activation checkpointing to be disabled.
It also requires specifying path to tokenizer's vocab file and merges file. 
Below example shows how to configure and use `megatron_generate` method for Megatron-LM GPT model.
```python
# specifying tokenizer's vocab and merges file
vocab_file = os.path.join(args.resume_from_checkpoint, "vocab.json")
merge_file = os.path.join(args.resume_from_checkpoint, "merges.txt")
other_megatron_args = {"vocab_file": vocab_file, "merge_file": merge_file}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)

# inference using `megatron_generate` functionality
tokenizer.pad_token = tokenizer.eos_token
max_new_tokens = 64
batch_texts = [
    "Are you human?",
    "The purpose of life is",
    "The arsenal was constructed at the request of",
    "How are you doing these days?",
]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)

# top-p sampling
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    top_p=0.8,
    top_p_decay=0.5,
    temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# top-k sampling
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    top_k=50,
    temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# adding `bos` token at the start
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"], batch_encodings["attention_mask"], max_new_tokens=max_new_tokens, add_BOS=True
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# beam search => only takes single prompt
batch_texts = ["The purpose of life is"]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    num_beams=20,
    length_penalty=1.5,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)
```

3. An end-to-end example of using `megatron_generate` method for Megatron-LM GPT model is available at
[megatron_gpt2_generation.py](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/inference/megatron_gpt2_generation.py) with 
config file [megatron_lm_gpt_generate_config.yaml](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/Configs/megatron_lm_gpt_generate_config.yaml).
The bash script with accelerate launch command is available at [megatron_lm_gpt_generate.sh](https://github.com/pacman100/accelerate-megatron-test/blob/main/megatron_lm_gpt_generate.sh).
The output logs of the script are available at [megatron_lm_gpt_generate.log](https://github.com/pacman100/accelerate-megatron-test/blob/main/output_logs/megatron_lm_gpt_generate.log).

## Support for ROPE and ALiBi Positional embeddings and Multi-Query Attention

1. For ROPE/ALiBi attention, pass `position_embedding_type` with `("absolute" | "rotary" | "alibi")` to `MegatronLMPlugin` as shown below.
```python
other_megatron_args = {"position_embedding_type": "alibi"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
```

2. For Multi-Query Attention, pass `attention_head_type` with `("multihead" | "multiquery")` to `MegatronLMPlugin` as shown below.
```python
other_megatron_args = {"attention_head_type": "multiquery"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
```

## Caveats

1. Supports Transformers GPT2, Megatron-BERT and T5 models.
This covers Decoder only, Encode only and Encoder-Decoder model classes.

2. Only loss is returned from model forward pass as 
there is quite complex interplay of pipeline, tensor and data parallelsim behind the scenes.
The `model(**batch_data)` call return loss(es) averaged across the data parallel ranks.
This is fine for most cases wherein pre-training jobs are run using Megatron-LM features and
you can easily compute the `perplexity` using the loss. 
For GPT model, returning logits in addition to loss(es) is supported. 
These logits aren't gathered across data prallel ranks. Use `accelerator.utils.gather_across_data_parallel_groups`
to gather logits across data parallel ranks. These logits along with labels can be used for computing various 
performance metrics. 

3. The main process is the last rank as the losses/logits are available in the last stage of pipeline.
`accelerator.is_main_process` and `accelerator.is_local_main_process` return `True` for last rank when using 
Megatron-LM integration.

4. In `accelerator.prepare` call, a Megatron-LM model corresponding to a given Transformers model is created
with random weights. Please use `accelerator.load_state` to load the Megatron-LM checkpoint with matching TP, PP and DP partitions.

5. Currently, checkpoint reshaping and interoperability support is only available for GPT. 
Soon it will be extended to BERT and T5.

6. `gradient_accumulation_steps` needs to be 1. When using Megatron-LM, micro batches in pipeline parallelism 
setting is synonymous with gradient accumulation. 

7. When using Megatron-LM, use `accelerator.save_state` and `accelerator.load_state` for saving and loading checkpoints.

8. Below are the mapping from Megatron-LM model architectures to the the equivalent 🤗 transformers model architectures.
Only these 🤗 transformers model architectures are supported.

a. Megatron-LM [BertModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/bert_model.py) : 
🤗 transformers models with `megatron-bert` in config's model type, e.g., 
[MegatronBERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)
    
b. Megatron-LM [GPTModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py) : 
🤗 transformers models with `gpt2` in config's model type, e.g., 
[OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
   
c. Megatron-LM [T5Model](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/t5_model.py) : 
🤗 transformers models with `t5` in  config's model type, e.g., 
[T5](https://huggingface.co/docs/transformers/model_doc/t5) and 
[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)