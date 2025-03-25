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

import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the memory usage data")
    parser.add_argument(
        "--memory_threshold",
        type=int,
        default=0,
        help="Memory threshold to filter data that is below this value (only filters 1st `--filter_partition` of the points which should roughtly correspond to the model loading)",
    )
    parser.add_argument(
        "--filter_partition",
        type=float,
        default=1 / 3,
        help="Partition to drop data from that are below the memory threshold",
    )
    return parser.parse_args()


def filter_data(data, memory_threshold, filter_partition, key):
    timestamps = data["timestamps"]
    memory = data[key]

    mid_point = int(len(timestamps) * filter_partition)
    filtered_times = []
    filtered_memory = []
    for i, (t, m) in enumerate(zip(timestamps, memory)):
        if i < mid_point and m < memory_threshold:
            continue
        filtered_times.append(t)
        filtered_memory.append(m)
    return filtered_times, filtered_memory


def compare_memory_usage(data, labels, memory_threshold, filter_partition):
    plt.style.use("seaborn-v0_8")
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]

    fig1, ax1 = plt.subplots(figsize=(15, 5))
    for data_item, label, color in zip(data, labels, colors):
        timestamps, allocated = filter_data(data_item, memory_threshold, filter_partition, "allocated_memory")
        ax1.plot(timestamps, allocated, label=label, color=color, linewidth=2)

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Allocated Memory (GB)", fontsize=12)
    ax1.set_title("Allocated Memory Usage Over Time", fontsize=14, pad=15)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(15, 5))
    for data_item, label, color in zip(data, labels, colors):
        timestamps, reserved = filter_data(data_item, memory_threshold, filter_partition, "reserved_memory")
        ax2.plot(timestamps, reserved, label=label, color=color, linewidth=2)

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Reserved Memory (GB)", fontsize=12)
    ax2.set_title("Reserved Memory Usage Over Time", fontsize=14, pad=15)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    return fig1, fig2


if __name__ == "__main__":
    args = parse_args()
    DIR = args.dir
    with open(f"{DIR}/torch_post_shard_memory_usage.json") as f:
        post_shard = json.load(f)

    with open(f"{DIR}/torch_pre_shard_not_fixed_memory_usage.json") as f:
        pre_shard = json.load(f)

    with open(f"{DIR}/torch_pre_shard_fixed_memory_usage.json") as f:
        pre_shard_fixed = json.load(f)

    with open(f"{DIR}/accelerate_memory_usage.json") as f:
        accelerate = json.load(f)

    data = [post_shard, pre_shard, pre_shard_fixed, accelerate]
    labels = [
        "Optimizer Post Sharding",
        "Optimizer Pre Sharding (w/o fix)",
        "Optimizer Pre Sharding (w/ fix)",
        "Accelerate",
    ]

    fig1, fig2 = compare_memory_usage(data, labels, args.memory_threshold, args.filter_partition)
    fig1.savefig(f"{DIR}/allocated_memory.png")
    fig2.savefig(f"{DIR}/reserved_memory.png")
