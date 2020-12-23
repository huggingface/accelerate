import torch

from .config import AcceleratorState, DistributedType, is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm


class AcceleratedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.state = AcceleratorState()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.state.distributed_type == DistributedType.TPU:
            xm.optimizer_step(self.optimizer)
        elif self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _switch_parameters(self, parameters_map):
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]
