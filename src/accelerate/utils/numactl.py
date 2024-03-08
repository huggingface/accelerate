# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
import subprocess
import sys

from accelerate.utils.environment import _get_nvidia_smi


def get_query_command(device_id: int = None, backend: str = None):
    """
    Returns the right param args for `numa` support based on a `backend`.

    Currently only supports `nvidia`.
    """
    if backend == "nvidia":
        return [_get_nvidia_smi(), "--query-gpu=pci.bus_id", "-i", str(device_id), "--format=csv,noheader"]
    else:
        # TODO: Add support for AMD and Gaudi2
        raise ValueError(f"Backend {backend} is currently not supported through `accelerate launch`.")


def get_bus_id(device_id: int = None, backend: str = None):
    """
    Gets the bus ID for `device_id`. If not passed, will default to the local rank.
    """
    if device_id is None:
        device_id = os.environ.get("LOCAL_RANK", 0)
    command = get_query_command(device_id, backend)
    output = subprocess.check_output(command, universal_newlines=True)
    return output.strip()[4:]


def get_numa_node_for_device(device_id: int = None):
    """
    Gets the numa node for `device_id`. If not passed, will default to the local rank.
    """
    bus_id = get_bus_id(device_id)
    if bus_id is None:
        raise ValueError(f"Could not find bus ID for device {device_id}.")

    with open(f"/sys/bus/pci/devices/{bus_id}/numa_node") as f:
        return f.read().strip()


def launch():
    args = sys.argv
    script, *script_args = args[1:]
    node = get_numa_node_for_device()
    cmd = ["numactl", f"--cpunodebind={node}", f"--membind={node}", "python3", script] + script_args
    process = subprocess.Popen(cmd, env=os.environ.copy())
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    launch()
