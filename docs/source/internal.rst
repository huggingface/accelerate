.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Internals
=======================================================================================================================


Optimizer
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: accelerate.optimizer.AcceleratedOptimizer


DataLoader
-----------------------------------------------------------------------------------------------------------------------

The main work on your PyTorch :obj:`DataLoader` is done by the following function:

.. autofunction:: accelerate.data_loader.prepare_data_loader


BatchSamplerShard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: accelerate.data_loader.DataLoaderShard
    :members:


BatchSamplerShard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: accelerate.data_loader.BatchSamplerShard
    :members:


IterableDatasetShard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: accelerate.data_loader.IterableDatasetShard
    :members:


Distributed Config
-----------------------------------------------------------------------------------------------------------------------


AcceleratorState
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: accelerate.state.AcceleratorState
    :members:


DistributedType
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: accelerate.state.DistributedType
    :members:


Utilities
-----------------------------------------------------------------------------------------------------------------------

.. autofunction:: accelerate.utils.extract_model_from_parallel

.. autofunction:: accelerate.utils.gather

.. autofunction:: accelerate.utils.send_to_device

.. autofunction:: accelerate.utils.set_seed

.. autofunction:: accelerate.utils.synchronize_rng_states

.. autofunction:: accelerate.utils.wait_for_everyone
