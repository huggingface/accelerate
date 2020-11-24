import torch

from packaging import version

from .config import DistributedState, DistributedType
from .data_loader import prepare_data_loader
from .optimizer import AcceleratedOptimizer
from .utils import extract_model_from_parallel


class Accelerator:
    def __init__(
        self,
        fp16: bool = None,
        put_objects_on_device: bool = False,
        split_batches_across_devices: bool = False,
    ):
        """
        Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

        Args:
            fp16 (:obj:`bool`, `optional`):
                Whether or not to use mixed precision training. Will default to the value in the environment variable
                :obj:`USE_FP16`, which will use the default value in the accelerate config of the current system or the
                flag passed with the :obj:`accelerate.launch` command.
            put_objects_on_device (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the accelerator should put objects on device (tensors yielded by the datalaoder, model,
                etc...).
            split_batches_across_devices (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the accelerator should split the batches yielded by the datalaoders across the devices.
                If :obj:`True` the actual batch size used will be the same on any kind of distributed processes, but it
                must be a round multiple of the :obj:`num_processes` you are using. If :obj:`False`, actual batch size
                used will be the one set in your script multiplied by the number of processes.
        """
        self.distributed_state = DistributedState()
        self.put_objects_on_device = put_objects_on_device
        self.split_batches_across_devices = split_batches_across_devices

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        self.fp16 = fp16 if fp16 is not None else self.distributed_state.use_fp16
        if fp16:
            self.native_amp = version.parse(torch.__version__) >= version.parse("1.6")
            self.scaler = torch.cuda.amp.GradScaler()

        # Internal references to the training objects
        self._optimizers = []

    @property
    def distributed_type(self):
        return self.distributed_state.distributed_type

    @property
    def num_processes(self):
        return self.distributed_state.num_processes

    @property
    def process_index(self):
        return self.distributed_state.process_index

    @property
    def local_process_index(self):
        return self.distributed_state.local_process_index

    @property
    def device(self):
        return self.distributed_state.device

    def is_main_process(self):
        return self.process_index == 0

    def is_main_local_process(self):
        return self.local_process_index == 0

    def print(self, *args, **kwargs):
        if self.is_main_local_process():
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
                    "the flag `put_objects_on_device = True` in your `Accelerator` creation to let it handle that "
                    "part for you."
                )

        # If we're dealing with device placement, this deals with that by...
        tpu_should_fix_optimizer = self.put_objects_on_device and self.distributed_type == DistributedType.TPU
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
        if self.put_objects_on_device:
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
            split_batches_across_devices=self.split_batches_across_devices,
            put_on_device=self.put_objects_on_device,
        )

    def prepare_optimizer(self, optimizer):
        return AcceleratedOptimizer(optimizer, scaler=self.scaler)

    def backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
        if self.fp16 and self.native_amp:
            for optimizer in self._optimizers:
                self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        # TODO: this unscales all optimizers where we should only unscale the one where parameters are.
        if self.fp16 and self.native_amp:
            for optimizer in self._optimizers:
                self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

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
