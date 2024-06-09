import unittest

from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    device_count,
    execute_subprocess_async,
    patch_environment,
    path_in_accelerate_package,
    require_multi_device,
)


class DDPCommHookTester(unittest.TestCase):
    test_file_path = path_in_accelerate_package("test_utils", "scripts", "test_ddp_comm_hook.py")

    @require_multi_device
    def test_ddp_communication_hooks(self):
        print(f"Found {device_count()} devices.")
        cmd = DEFAULT_LAUNCH_COMMAND + [self.test_file_path]
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)


if __name__ == "__main__":
    unittest.main()
