# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from accelerate.utils import get_torch_accelerator_backend


def main():
    device_type, backend = get_torch_accelerator_backend()
    num_devices = 1
    if backend is not None:
        num_devices = backend.device_count()

    if device_type is None:
        device_name = "CPU"
    elif device_type == "cuda":
        device_name = "GPU"
    else:
        device_name = device_type.upper()
    print(f"Successfully ran on {num_devices} {device_name}(s)")


if __name__ == "__main__":
    main()
