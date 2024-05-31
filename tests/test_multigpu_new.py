import os
import torch
import inspect
import unittest

from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import run_tests

from accelerate import Accelerator, PartialState
from accelerate.test_utils import device_count

class TrainingTester(MultiThreadedTestCase):
    @property
    def world_size(self):
        return device_count

    def setUp(self):
        super().setUp()
        self._spawn_threads()
    
    # Verify we are running in multiproc
    def test_distributed_spawning(self):
        state = PartialState()
        assert state.local_process_index == torch.distributed.get_rank()
        assert state.num_processes == torch.distributed.get_world_size()

if __name__ == "__main__":
    run_tests()