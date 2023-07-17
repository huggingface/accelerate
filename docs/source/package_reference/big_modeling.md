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

# Working with large models

## Dispatching and Offloading Models

[[autodoc]] big_modeling.init_empty_weights
[[autodoc]] big_modeling.cpu_offload
[[autodoc]] big_modeling.disk_offload
[[autodoc]] big_modeling.dispatch_model
[[autodoc]] big_modeling.load_checkpoint_and_dispatch
[[autodoc]] big_modeling.load_checkpoint_in_model
[[autodoc]] utils.infer_auto_device_map

## Model Hooks

### Hook Classes

[[autodoc]] hooks.ModelHook
[[autodoc]] hooks.AlignDevicesHook
[[autodoc]] hooks.SequentialHook

### Adding Hooks

[[autodoc]] hooks.add_hook_to_module
[[autodoc]] hooks.attach_execution_device_hook
[[autodoc]] hooks.attach_align_device_hook
[[autodoc]] hooks.attach_align_device_hook_on_blocks

### Removing Hooks

[[autodoc]] hooks.remove_hook_from_module
[[autodoc]] hooks.remove_hook_from_submodules