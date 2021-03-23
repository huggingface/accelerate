# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import copy
from dataclasses import dataclass


class KwargsHandler:
    """
    Internal mixin that implements a :obj:`to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your :class:`~accelerate.Accelerator` to customize how your model is wrapped in a
    :obj:`torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this `wrapper
    <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ for more information
    on each argument.

    .. warning::

        :obj:`gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your :class:`~accelerate.Accelerator` to customize the behavior of mixed precision, specifically
    how the :obj:`torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this `scaler
    <https://pytorch.org/docs/stable/amp.html?highlight=gradscaler>`__ for more information on each argument.

    .. warning::

        :obj:`GradScaler` is only available in PyTorch 1.5.0 and later versions.
    """

    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True
