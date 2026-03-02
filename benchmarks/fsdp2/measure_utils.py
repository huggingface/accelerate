# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import os
import threading
import time

import psutil
import torch

from accelerate import PartialState


class MemoryTracker:
    def __init__(
        self,
        device: torch.device,
        output_directory: str,
        run_name: str,
        save_memory_snapshot: bool,
        log_interval: float = 0.01,
    ):
        """Class for tracking gpu and cpu memory usage of the process.

        Args:
            device (`torch.device`):
                PyTorch device to monitor.
            output_directory (`str`):
                Directory to save the memory usage data to, will be created if it doesn't exist.
            run_name (`str`):
                Name of the run, will be used to name the output files.
            save_memory_snapshot (`bool`):
                Whether to also save `torch.cuda.memory._dump_snapshot` to the output directory.
            log_interval (`float`, *optional*):
                Interval in seconds between memory measurements. Defaults to 0.01.
        """
        self.log_interval = log_interval
        self.save_memory_snapshot = save_memory_snapshot
        self.output_directory = output_directory
        self.run_name = run_name

        self.timestamps = []
        self.allocated_memory = []
        self.reserved_memory = []
        self.virtual_memory = []

        self.start_time = None
        self.running = False

        self._thread = None
        self._state = PartialState()
        self._process = psutil.Process()
        self._device = device
        self.torch_accelerator_module = getattr(torch, device.type, torch.cuda)

    def _monitor(self):
        self.start_time = time.time()

        while self.running:
            allocated = self.torch_accelerator_module.memory_allocated(self._device) / (1024 * 1024)
            reserved = self.torch_accelerator_module.memory_reserved(self._device) / (1024 * 1024)
            virtual_memory = self._process.memory_info().rss / (1024 * 1024)

            self.allocated_memory.append(allocated)
            self.reserved_memory.append(reserved)
            self.virtual_memory.append(virtual_memory)
            self.timestamps.append(time.time() - self.start_time)

            time.sleep(self.log_interval)

    def start(self):
        gc.collect()
        self.torch_accelerator_module.empty_cache()

        if self.output_directory:
            os.makedirs(self.output_directory, exist_ok=True)

        if self.save_memory_snapshot:
            self.torch_accelerator_module.memory._record_memory_history()

        self.running = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

        if self.save_memory_snapshot and self._state.is_main_process and self.output_directory:
            output_file = os.path.join(self.output_directory, f"{self.run_name}_memory_snapshot.pkl")
            self.torch_accelerator_module.memory._dump_snapshot(output_file)

        if self._state.is_main_process and self.output_directory:
            path = os.path.join(self.output_directory, f"{self.run_name}_memory_usage.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "timestamps": self.timestamps,
                        "allocated_memory": self.allocated_memory,
                        "reserved_memory": self.reserved_memory,
                        "virtual_memory": self.virtual_memory,
                    },
                    f,
                )
        if self.save_memory_snapshot:
            self.torch_accelerator_module.memory._record_memory_history(False)
        self.torch_accelerator_module.empty_cache()

    @property
    def peak_allocated_memory(self):
        return max(self.allocated_memory)

    @property
    def peak_reserved_memory(self):
        return max(self.reserved_memory)
