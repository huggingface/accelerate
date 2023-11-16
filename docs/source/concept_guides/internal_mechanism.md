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

# 🤗 Accelerate's internal mechanisms

Internally, 🤗 Accelerate works by first analyzing the environment in which the script is launched to determine which
kind of distributed setup is used, how many different processes there are and which one the current script is in. All
that information is stored in the [`~AcceleratorState`].

This class is initialized the first time you instantiate an [`~Accelerator`] as well as performing any
specific initialization your distributed setup needs. Its state is then uniquely shared through all instances of
[`~state.AcceleratorState`]. (The same can also be done with the [`PartialState`], a more barebones version it inherits)

Then, when calling [`~Accelerator.prepare`], the library:

- wraps your model(s) in the container adapted for the distributed setup,
- wraps your optimizer(s) in an [`~optimizer.AcceleratedOptimizer`],
- wraps your scheduler(s) in an [`~scheduler.AcceleratedScheduler`]
- creates a new version of your dataloader(s) in a [`~data_loader.DataLoaderShard`] or [`~data_loader.DataLoaderDispatcher`]

While the model(s), optimizer(s), and scheduler(s) are just put in simple wrappers, the dataloader(s) are re-created. This is mostly
because PyTorch does not let the user change the `batch_sampler` of a dataloader once it's been created and the
library handles the sharding of your data between processes by changing that `batch_sampler` to yield every other
`num_processes` batches (if enabled).

The [`~data_loader.DataLoaderShard`] subclasses `DataLoader` to add the following functionality:

- it synchronizes the appropriate random number generator of all processes at each new iteration, to ensure any
  randomization (like shuffling) is done the exact same way across processes.
- it puts the batches on the proper device before yielding them (unless you have opted out of
  `device_placement=True`).
  
The [`~data_loader.DataLoaderDispatcher`] subclasses differs from the [`~data_loader.DataLoaderShard`] in that when iterating through the `DataLoader`, the data is all starting from process 0 and *then* split and sent off to each process rather than it happening at the dataset level.

The random number generator synchronization will by default synchronize:

- the `generator` attribute of a given sampler (like the PyTorch `RandomSampler`) for PyTorch >= 1.6
- the main random number generator in PyTorch <=1.5.1

You can choose which random number generator(s) to synchronize with the `rng_types` argument of the main
[`Accelerator`]. In PyTorch >= 1.6, it is recommended to rely on a local `generator` to avoid
setting the same seed in the main random number generator in all processes.

<Tip warning={true}>

    Synchronization of the main torch (or CUDA or XLA) random number generator will affect any other potential random
    artifacts you could have in your dataset (like random data augmentation) in the sense that all processes will get
    the same random numbers from the torch random modules (so will apply the same random data augmentation if it's
    controlled by torch).

</Tip>

<Tip>

    The randomization part of your custom sampler, batch sampler or iterable dataset should be done using a local
    `torch.Generator` object (in PyTorch >= 1.6), see the traditional `RandomSampler`, as an example.

</Tip>

For more details about the internals, see the [Internals page](package_reference/torch_wrappers).
