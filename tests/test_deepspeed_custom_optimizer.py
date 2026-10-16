# Copyright 2026 The HuggingFace Team. All rights reserved.
import torch
from torch import nn

from accelerate.utils.deepspeed import is_stock_torch_optimizer


class _CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, dict(lr=lr))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        return loss


def test_is_stock_torch_optimizer_true_for_adamw():
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert is_stock_torch_optimizer(opt)


def test_is_stock_torch_optimizer_false_for_custom_class():
    model = nn.Linear(4, 2)
    opt = _CustomOptimizer(model.parameters(), lr=1e-3)
    assert not is_stock_torch_optimizer(opt)


def test_is_stock_torch_optimizer_false_for_non_optimizer():
    assert not is_stock_torch_optimizer(nn.Linear(2, 2))
