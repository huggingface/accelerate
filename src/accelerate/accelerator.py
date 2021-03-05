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

import torch

from packaging import version

from .data_loader import prepare_data_loader
from .optimizer import AcceleratedOptimizer
from .state import AcceleratorState, DistributedType
from .utils import extract_model_from_parallel, gather, save, wait_for_everyone


class Accelerator:
    """
    Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

    Args:
        device_placement (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
            etc...).
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            :obj:`True` the actual batch size used will be the same on any kind of distributed processes, but it must
            be a round multiple of the :obj:`num_processes` you are using. If :obj:`False`, actual batch size used will
            be the one set in your script multiplied by the number of processes.
        fp16 (:obj:`bool`, `optional`):
            Whether or not to use mixed precision training. Will default to the value in the environment variable
            :obj:`USE_FP16`, which will use the default value in the accelerate config of the current system or the
            flag passed with the :obj:`accelerate.launch` command.
        cpu (:obj:`bool`, `optional`):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to :obj:`True` and
            force the execution on one process only.

    Attributes

        - **device** (:obj:`torch.device`) -- The device to use.
        - **state** (:class:`~accelerate.AcceleratorState`) -- The distributed setup state.
    """

    def __init__(
        self, device_placement: bool = True, split_batches: bool = False, fp16: bool = None, cpu: bool = False
    ):
        self.state = AcceleratorState(fp16=fp16, cpu=cpu, _from_accelerator=True)

        self.device_placement = device_placement
        self.split_batches = split_batches

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if self.state.use_fp16:
            self.native_amp = version.parse(torch.__version__) >= version.parse("1.6")
            self.scaler = torch.cuda.amp.GradScaler()

        # Internal references to the training objects
        self._optimizers = []

    @property
    def distributed_type(self):
        return self.state.distributed_type

    @property
    def num_processes(self):
        return self.state.num_processes

    @property
    def process_index(self):
        return self.state.process_index

    @property
    def local_process_index(self):
        return self.state.local_process_index

    @property
    def device(self):
        return self.state.device

    @property
    def is_main_process(self):
        """True for one process only."""
        return self.process_index == 0

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.local_process_index == 0

    @property
    def use_fp16(self):
        return self.state.use_fp16

    def print(self, *args, **kwargs):
        """
        Use in replacement of :obj:`print()` to only print once per server.
        """
        if self.is_local_main_process:
            print(*args, **kwargs)

    def _prepare_one(self, obj):
        if isinstance(obj, torch.utils.data.DataLoader):
            return self.prepare_data_loader(obj)
        elif isinstance(obj, torch.nn.Module):
            return self.prepare_model(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = self.prepare_optimizer(obj)
            self._optimizers.append(optimizer)
            return optimizer
        else:
            return obj

    def prepare(self, *args):
        """
        Prepare all objects passed in :obj:`args` for distributed training and mixed precision, then return them in the
        same order.

        Accepts the following type of objects:

            - :obj:`torch.utils.data.DataLoader`: PyTorch Dataloader
            - :obj:`torch.nn.Module`: PyTorch Module
            - :obj:`torch.optim.Optimizer`: PyTorch Optimizer

        """
        # On TPUs, putting the model on the XLA device will create new parameters, so the corresponding optimizer will
        # have parameters disconnected from the model (so no training :-( ).
        # If the model and optimizer have parameters on different devices we raise an error.
        if self.distributed_type == DistributedType.TPU:
            model_device, optimizer_device = self._get_devices()
            if model_device is not None and optimizer_device is not None and model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you "
                    "created an optimizer around your model **before** putting on the device. Make sure the line "
                    "model.to(device) is before the optimizer creation in your script or remove it entirely and use "
                    "the flag default value for `devicement_placement` in your `Accelerator` to let it handle that "
                    "part for you."
                )

        # If we're dealing with device placement, this deals with that by...
        tpu_should_fix_optimizer = self.device_placement and self.distributed_type == DistributedType.TPU
        if tpu_should_fix_optimizer:
            # 1. grabbing old model parameters
            old_named_params = self._get_named_parameters(*args)

        result = tuple(self._prepare_one(obj) for obj in args)

        if tpu_should_fix_optimizer:
            # 2. grabbing new model parameters
            new_named_params = self._get_named_parameters(*result)
            # 3. building a map from the first to the second
            mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
            # 4. using that map to update the parameters of the optimizer
            for obj in result:
                if isinstance(obj, torch.optim.Optimizer):
                    obj._switch_parameters(mapping)

        return result

    def prepare_model(self, model):
        if self.device_placement:
            model = model.to(self.device)
        if self.distributed_type == DistributedType.MULTI_GPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_process_index],
                output_device=self.local_process_index,
            )
        if self.native_amp:
            model.forward = torch.cuda.amp.autocast()(model.forward)
        return model

    def prepare_data_loader(self, data_loader):
        return prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=self.device_placement,
        )

    def prepare_optimizer(self, optimizer):
        return AcceleratedOptimizer(optimizer, device_placement=self.device_placement, scaler=self.scaler)

    def backward(self, loss):
        """
        Use :obj:`accelerator.backward(loss)` in lieu of :obj:`loss.backward()`.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """
        Should be used in place of :func:`torch.nn.utils.clip_grad_norm_`.
        """
        # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
        if self.state.use_fp16 and self.native_amp:
            for optimizer in self._optimizers:
                self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        """
        Should be used in place of :func:`torch.nn.utils.clip_grad_value_`.
        """
        # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
        if self.state.use_fp16 and self.native_amp:
            for optimizer in self._optimizers:
                self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    def gather(self, tensor):
        """
        Gather the values in `tensor` accross all processes and concatenate them on the first dimension. Useful to
        regroup the predictions from all processes when doing evaluation.

        Note:
            This gather happens in all processes.

        Args:
            tensor (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`):
                The tensors to gather accross all processes.

        Returns:
            :obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`: The gathered tensor(s). Note
            that the first dimension of the result is `num_processes` multiplied by the first dimension of the input
            tensors.
        """
        return gather(tensor)

    def unwrap_model(self, model):
        """
        Unwraps the :obj:`model` from the additional layer possible added by :meth:`~accelerate.Accelerator.prepare`.
        Useful before saving the model.

        Args:
            model (:obj:`torch.nn.Module`):
                The model to unwrap.
        """
        return extract_model_from_parallel(model)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.
        """
        wait_for_everyone()

    def save(self, obj, f):
        """
        Save the object passed to disk once per machine. Use in place of :obj:`torch.save`.

        Args:
            obj: The object to save.
            f (:obj:`str` or :obj:`os.PathLike`):
                Where to save the content of :obj:`obj`.
        """
        save(obj, f)

    def _get_named_parameters(self, *args):
        named_parameters = {}
        for obj in args:
            if isinstance(obj, torch.nn.Module):
                obj = extract_model_from_parallel(obj)
                named_parameters.update({n: p for n, p in obj.named_parameters()})
        return named_parameters

    def _get_devices(self, *args):
        model_device = None
        optimizer_device = None
        for obj in args:
            # Loop through model parameters and stop at the first once we have its device.
            if isinstance(obj, torch.nn.Module):
                for param in obj.parameters():
                    model_device = param.device
                    break
            # Loop through optimizer parameters groups and stop at the first once we have its device.
            if isinstance(obj, torch.optim.Optimizer):
                for param_group in obj.param_groups:
                    if len(param_group["params"]) > 0:
                        optimizer_device = param_group["params"][0].device
                        break
        return (model_device, optimizer_device)
