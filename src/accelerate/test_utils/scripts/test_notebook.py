import torch


assert not torch.cuda.is_initialized(), "CUDA was initialized before the test script."


assert not torch.cuda.is_initialized(), "CUDA was initialized upon importing the `Accelerator` class."
