import inspect
import os
import sys
import unittest

import torch

import accelerate
from accelerate.test_utils import execute_subprocess_async, require_multi_gpu


class MultiGPUTester(unittest.TestCase):
    def setUp(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["test_script.py"])

    @require_multi_gpu
    def test_multi_gpu(self):
        print(f"Found {torch.cuda.device_count()} devices.")
        distributed_args = f"""
            -m torch.distributed.launch
            --nproc_per_node={torch.cuda.device_count()}
            --use_env
            {self.test_file_path}
        """.split()
        cmd = [sys.executable] + distributed_args
        execute_subprocess_async(cmd, env=os.environ.copy())
