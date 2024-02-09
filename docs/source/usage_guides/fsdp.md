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

For instance, here is how you would run `examples/nlp_example.py` (from the root of the repo) with FSDP enabled:

```bash
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py
```

Currently, `Accelerate` supports the following config through the CLI:

`fsdp_sharding_strategy`: [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy), [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy). For more information, please refer the official [PyTorch docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy).

`fsdp_offload_params` : Decides Whether to offload parameters and gradients to CPU

`fsdp_auto_wrap_policy`: [1] TRANSFORMER_BASED_WRAP, [2] SIZE_BASED_WRAP, [3] NO_WRAP

`fsdp_transformer_layer_cls_to_wrap`: Only applicable for 🤗 Transformers. When using `fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP`, a user may provide a comma-separated string of transformer layer class names (case-sensitive) to wrap, e.g., `BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput`. This is important because submodules that share weights (e.g., embedding layers) should not end up in different FSDP wrapped units. Using this policy, wrapping happens for each block containing Multi-Head Attention followed by a couple of MLP layers. Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit. Therefore, use this for transformer-based models. You can use the `model._no_split_modules` for 🤗 Transformer models by answering `yes` to `Do you want to use the model's `_no_split_modules` to wrap. It will try to use `model._no_split_modules` when possible.

`fsdp_min_num_params`: minimum number of parameters when using `fsdp_auto_wrap_policy=SIZE_BASED_WRAP`.

`fsdp_backward_prefetch_policy`: [1] BACKWARD_PRE, [2] BACKWARD_POST, [3] NO_PREFETCH

`fsdp_forward_prefetch`: if True, then FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass. Should only be used for static-graph models since the prefetching follows the first iteration’s execution order. i.e., if the sub-modules' order changes dynamically during the model's execution do not enable this feature.

`fsdp_state_dict_type`: [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT

`fsdp_use_orig_params`: If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable parameters. This setting is useful in cases such as parameter-efficient fine-tuning as discussed in [this post](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019). This option also allows one to have multiple optimizer param groups. This should be `True` when creating an optimizer before preparing/wrapping the model with FSDP.

`fsdp_cpu_ram_efficient_loading`: Only applicable for 🤗 Transformers models. If True, only the first process loads the pretrained model checkpoint while all other processes have empty weights. This should be set to False if you experience errors when loading the pretrained 🤗 Transformers model via `from_pretrained` method. When this setting is True `fsdp_sync_module_states` also must to be True, otherwise all the processes except the main process would have random weights leading to unexpected behaviour during training. For this to work, make sure the distributed process group is initialized before calling Transformers `from_pretrained` method. When using 🤗 Trainer API, the distributed process group is initialized when you create an instance of `TrainingArguments` class.

`fsdp_sync_module_states`: If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0.


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

Inspect the checkpoint folder to see model and optimizer as shards per process:
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

`accelerator.get_state_dict` will call the underlying `model.state_dict` implementation using `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` context manager to get the state dict only for rank 0 and it will be offloaded to CPU.

You can then pass `state` into the `save_pretrained` method.  There are several modes for `StateDictType` and `FullStateDictConfig` that you can use to control the behavior of `state_dict`.  For more information, see the [PyTorch documentation](https://pytorch.org/docs/stable/fsdp.html).


## Mapping between FSDP sharding strategies and DeepSpeed ZeRO Stages
* `FULL_SHARD` maps to the DeepSpeed `ZeRO Stage-3`. Shards optimizer states, gradients and parameters.
* `SHARD_GRAD_OP` maps to the DeepSpeed `ZeRO Stage-2`. Shards optimizer states and gradients.
* `NO_SHARD` maps to `ZeRO Stage-0`. No sharding wherein each GPU has full copy of model, optimizer states and gradients.
* `HYBRID_SHARD` maps to `ZeRO++ Stage-3` wherein `zero_hpz_partition_size=<num_gpus_per_node>`. Here, this will shard optimizer states, gradients and parameters within each node while each node has full copy.

## A few caveats to be aware of

- In case of multiple models, pass the optimizers to the prepare call in the same order as corresponding models else `accelerator.save_state()` and `accelerator.load_state()` will result in wrong/unexpected behaviour.
- This feature is incompatible with `--predict_with_generate` in the `run_translation.py` script of 🤗 `Transformers` library.

For more control, users can leverage the `FullyShardedDataParallelPlugin`. After creating an instance of this class, users can pass it to the Accelerator class instantiation.
For more information on these options, please refer to the PyTorch [FullyShardedDataParallel](https://github.com/pytorch/pytorch/blob/0df2e863fbd5993a7b9e652910792bd21a516ff3/torch/distributed/fsdp/fully_sharded_data_parallel.py#L236) code.
