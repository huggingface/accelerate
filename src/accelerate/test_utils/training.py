import numpy as np
import torch


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.y = a * self.x + b + np.random.normal(scale=0.1, size=(length,)).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class RegressionModel(torch.nn.Module):
    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())

    def forward(self, x=None):
        return x * self.a + self.b
