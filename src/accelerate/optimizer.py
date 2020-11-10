import torch

from .config import DistributedState, DistributedType, is_tpu_available

if is_tpu_available():
    import torch_xla.core.xla_model as xm

class AcceleratedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = DistributedState()
    
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
        else:
            self.optimizer.step()   
