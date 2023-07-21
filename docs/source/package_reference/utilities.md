<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Helpful Utilities

Below are a variety of utility functions that 🤗 Accelerate provides, broken down by use-case. 

## Constants

Constants used throughout 🤗 Accelerate for reference

The following are constants used when utilizing [`Accelerator.save_state`]

`utils.MODEL_NAME`: `"pytorch_model"`
`utils.OPTIMIZER_NAME`: `"optimizer"`
`utils.RNG_STATE_NAME`: `"random_states"`
`utils.SCALER_NAME`: `"scaler.pt`
`utils.SCHEDULER_NAME`: `"scheduler`

The following are constants used when utilizing [`Accelerator.save_model`]

`utils.WEIGHTS_NAME`: `"pytorch_model.bin"`
`utils.SAFE_WEIGHTS_NAME`: `"model.safetensors"`
`utils.WEIGHTS_INDEX_NAME`: `"pytorch_model.bin.index.json"`
`utils.SAFE_WEIGHTS_INDEX_NAME`: `"model.safetensors.index.json"`

## Data Classes

These are basic dataclasses used throughout 🤗 Accelerate and they can be passed in as parameters.

[[autodoc]] utils.DistributedType

[[autodoc]] utils.DynamoBackend

[[autodoc]] utils.LoggerType

[[autodoc]] utils.PrecisionType

[[autodoc]] utils.ProjectConfiguration

## Plugins

These are plugins that can be passed to the [`Accelerator`] object. While they are defined elsewhere in the documentation, 
for convience all of them are available to see here:

[[autodoc]] utils.DeepSpeedPlugin

[[autodoc]] utils.FullyShardedDataParallelPlugin

[[autodoc]] utils.GradientAccumulationPlugin

[[autodoc]] utils.MegatronLMPlugin

[[autodoc]] utils.TorchDynamoPlugin


## Data Manipulation and Operations

These include data operations that mimic the same `torch` ops but can be used on distributed processes.

[[autodoc]] utils.broadcast

[[autodoc]] utils.concatenate

[[autodoc]] utils.gather

[[autodoc]] utils.pad_across_processes

[[autodoc]] utils.reduce

[[autodoc]] utils.send_to_device

## Environment Checks

These functionalities check the state of the current working environment including information about the operating system itself, what it can support, and if particular dependencies are installed. 

[[autodoc]] utils.is_bf16_available

[[autodoc]] utils.is_ipex_available

[[autodoc]] utils.is_mps_available

[[autodoc]] utils.is_npu_available

[[autodoc]] utils.is_torch_version

[[autodoc]] utils.is_tpu_available

[[autodoc]] utils.is_xpu_available

## Environment Manipulation

[[autodoc]] utils.patch_environment

[[autodoc]] utils.clear_environment

[[autodoc]] utils.write_basic_config

When setting up 🤗 Accelerate for the first time, rather than running `accelerate config` [~utils.write_basic_config] can be used as an alternative for quick configuration.

## Memory

[[autodoc]] utils.get_max_memory

[[autodoc]] utils.find_executable_batch_size

## Modeling

These utilities relate to interacting with PyTorch models

[[autodoc]] utils.extract_model_from_parallel

[[autodoc]] utils.get_max_layer_size

[[autodoc]] utils.offload_state_dict


## Parallel

These include general utilities that should be used when working in parallel.

[[autodoc]] utils.extract_model_from_parallel

[[autodoc]] utils.save

[[autodoc]] utils.wait_for_everyone


## Random

These utilities relate to setting and synchronizing of all the random states.

[[autodoc]] utils.set_seed

[[autodoc]] utils.synchronize_rng_state

[[autodoc]] utils.synchronize_rng_states


## PyTorch XLA

These include utilities that are useful while using PyTorch with XLA.

[[autodoc]] utils.install_xla

## Loading model weights

These include utilities that are useful to load checkpoints.

[[autodoc]] utils.load_checkpoint_in_model

## Quantization

These include utilities that are useful to quantize model.

[[autodoc]] utils.load_and_quantize_model

[[autodoc]] utils.BnbQuantizationConfig