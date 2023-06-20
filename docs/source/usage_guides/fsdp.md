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

# Fully Sharded Data Parallel

To accelerate training huge models on larger batch sizes, we can use a fully sharded data parallel model.
This type of data parallel paradigm enables fitting more data and larger models by sharding the optimizer states, gradients and parameters.
To read more about it and the benefits, check out the [Fully Sharded Data Parallel blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/).
We have integrated the latest PyTorch's Fully Sharded Data Parallel (FSDP) training feature.
All you need to do is enable it through the config.

## How it works out of the box

On your machine(s) just run:

```bash
accelerate config
```

and answer the questions asked. This will generate a config file that will be used automatically to properly set the
default options when doing

```bash
accelerate launch my_script.py --args_to_my_script
```

For instance, here is how you would run the NLP example (from the root of the repo) with FSDP enabled:

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: GPT2Block
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py
```

Currently, `Accelerate` supports the following config through the CLI:

```bash
`Sharding Strategy`: [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD
`Offload Params`: Decides Whether to offload parameters and gradients to CPU
`Auto Wrap Policy`: [1] TRANSFORMER_BASED_WRAP, [2] SIZE_BASED_WRAP, [3] NO_WRAP [4] "HYBRID_SHARD" [5] "HYBRID_SHARD_ZERO2"
`Transformer Layer Class to Wrap`: When using `TRANSFORMER_BASED_WRAP`, user specifies comma-separated string of transformer layer class names (case-sensitive) to wrap ,e.g, 
`BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput`...
`Min Num Params`: minimum number of parameters when using `SIZE_BASED_WRAP`
`Backward Prefetch`: [1] BACKWARD_PRE, [2] BACKWARD_POST, [3] NO_PREFETCH
`State Dict Type`: [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT  
`Use Orig Params`: If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres. 
Useful in cases such as parameter-efficient fine-tuning. 
Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)
`Sync Module States`: If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0
`Forward Prefetch`: If True, then FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass
```

For additional and more nuanced control, you can specify other FSDP parameters via `FullyShardedDataParallelPlugin`. 
When creating `FullyShardedDataParallelPlugin` object, pass it the parameters that weren't part of the accelerate config or if you want to override them.
The FSDP parameters will be picked based on the accelerate config file or launch command arguments and other parameters that you will pass directly through the `FullyShardedDataParallelPlugin` object will set/override that.

Below is an example:

```py
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

## Saving and loading

The new recommended way of checkpointing when using FSDP models is to use `SHARDED_STATE_DICT` as `StateDictType` when setting up the accelerate config.
Below is the code snippet to save using `save_state` utility of accelerate.

```py
accelerator.save_state("ckpt")
```

Inspect the ckeckpoint folder to see model and optimizer as shards per process:
```
ls ckpt 
# optimizer_0  pytorch_model_0  random_states_0.pkl  random_states_1.pkl  scheduler.bin

cd ckpt

ls optimizer_0
# __0_0.distcp  __1_0.distcp

ls pytorch_model_0
# __0_0.distcp  __1_0.distcp
```

To load them back for resuming the training, use the `load_state` utility of accelerate

```py
accelerator.load_state("ckpt")
```

When using transformers `save_pretrained`, pass `state_dict=accelerator.get_state_dict(model)` to save the model state dict. 
  Below is an example:

```diff
  unwrapped_model.save_pretrained(
      args.output_dir,
      is_main_process=accelerator.is_main_process,
      save_function=accelerator.save,
+     state_dict=accelerator.get_state_dict(model),
)
```

### State Dict

`accelerator.get_state_dict` will call the underlying `model.state_dict` implementation.  With a model wrapped by FSDP, the default behavior of `state_dict` is to gather all of the state in the rank 0 device.  This can cause CUDA out of memory errors if the parameters don't fit on a single GPU.

To avoid this, PyTorch provides a context manager that adjusts the behavior of `state_dict`.  To offload some of the state dict onto CPU, you can use the following code:

```
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state = accelerator.get_state_dict(unwrapped_model)
```

You can then pass `state` into the `save_pretrained` method.  There are several modes for `StateDictType` and `FullStateDictConfig` that you can use to control the behavior of `state_dict`.  For more information, see the [PyTorch documentation](https://pytorch.org/docs/stable/fsdp.html).

## A few caveats to be aware of

- PyTorch FSDP auto wraps sub-modules, flattens the parameters and shards the parameters in place.
  Due to this, any optimizer created before model wrapping gets broken and occupies more memory.
  Hence, it is highly recommended and efficient to prepare the model before creating the optimizer.
  `Accelerate` will automatically wrap the model and create an optimizer for you in case of single model with a warning message.
  > FSDP Warning: When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer

However, below is the recommended way to prepare model and optimizer while using FSDP:

```diff
  model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
+ model = accelerator.prepare(model)

  optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

- model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
-        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
-    )

+ optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
+         optimizer, train_dataloader, eval_dataloader, lr_scheduler
+    )
```

- In case of a single model, if you have created the optimizer with multiple parameter groups and called prepare with them together,
  then the parameter groups will be lost and the following warning is displayed:
  > FSDP Warning: When using FSDP, several parameter groups will be conflated into
  > a single one due to nested module wrapping and parameter flattening.
  
  This is because parameter groups created before wrapping will have no meaning post wrapping due to parameter flattening of nested FSDP modules into 1D arrays (which can consume many layers).
  For instance, below are the named parameters of an FSDP model on GPU 0 (When using 2 GPUs. Around 55M (110M/2) params in 1D arrays as this will have the 1st shard of the parameters). 
  Here, if one has applied no weight decay for [bias, LayerNorm.weight] the named parameters of an unwrapped BERT model, 
  it can't be applied to the below FSDP wrapped model as there are no named parameters with either of those strings and 
  the parameters of those layers are concatenated with parameters of various other layers.
  ```
  {
    '_fsdp_wrapped_module.flat_param': torch.Size([494209]), 
    '_fsdp_wrapped_module._fpw_module.bert.embeddings.word_embeddings._fsdp_wrapped_module.flat_param': torch.Size([11720448]), 
    '_fsdp_wrapped_module._fpw_module.bert.encoder._fsdp_wrapped_module.flat_param': torch.Size([42527232])
  }
  ```


- In case of multiple models, it is necessary to prepare the models before creating optimizers or else it will throw an error. 
Then pass the optimizers to the prepare call in the same order as corresponding models else `accelerator.save_state()` and `accelerator.load_state()` will result in wrong/unexpected behaviour.
- This feature is incompatible with `--predict_with_generate` in the `run_translation.py` script of 🤗 `Transformers` library.

For more control, users can leverage the `FullyShardedDataParallelPlugin`. After creating an instance of this class, users can pass it to the Accelerator class instantiation.
For more information on these options, please refer to the PyTorch [FullyShardedDataParallel](https://github.com/pytorch/pytorch/blob/0df2e863fbd5993a7b9e652910792bd21a516ff3/torch/distributed/fsdp/fully_sharded_data_parallel.py#L236) code.
