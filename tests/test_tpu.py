import inspect
import os
import sys
import unittest

import accelerate
from accelerate.test_utils import execute_subprocess_async, require_tpu


class MultiTPUTester(unittest.TestCase):
    def setUp(self):
        mod_file = inspect.getfile(accelerate.test_utils)
        self.test_file_path = os.path.sep.join(mod_file.split(os.path.sep)[:-1] + ["test_script.py"])
        self.test_dir = os.path.sep.join(inspect.getfile(self.__class__).split(os.path.sep)[:-1])

    @require_tpu
    def test_tpu(self):
        distributed_args = f"""
            {self.test_dir}/xla_spawn.py
            --num_cores 8
            {self.test_file_path}
        """.split()
        cmd = [sys.executable] + distributed_args
        execute_subprocess_async(cmd, env=os.environ.copy())
