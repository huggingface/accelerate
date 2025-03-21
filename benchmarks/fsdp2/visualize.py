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
    parser.add_argument("--dir", type=str, default="test_dir")
    parser.add_argument("--memory_threshold", type=int, default=0)
    return parser.parse_args()


def compare_memory_usage(data, labels, memory_threshold):
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    for data_item, label in zip(data, labels):
        timestamps = data_item["timestamps"]
        allocated = data_item["allocated_memory"]

        # Filter data: remove points below 500MB in first 50% of data
        mid_point = len(timestamps) // 2
        filtered_times = []
        filtered_allocated = []
        for i, (t, m) in enumerate(zip(timestamps, allocated)):
            if i < mid_point and m < memory_threshold:
                continue
            filtered_times.append(t)
            filtered_allocated.append(m)

        ax1.plot(filtered_times, filtered_allocated, label=label)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Allocated Memory")
    ax1.set_title("Allocated Memory")
    ax1.legend()
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(15, 5))
    for data_item, label in zip(data, labels):
        timestamps = data_item["timestamps"]
        reserved = data_item["reserved_memory"]

        # Filter data: remove points below 500MB in first 50% of data
        mid_point = len(timestamps) // 2
        filtered_times = []
        filtered_reserved = []
        for i, (t, m) in enumerate(zip(timestamps, reserved)):
            if i < mid_point and m < args.memory_threshold:
                continue
            filtered_times.append(t)
            filtered_reserved.append(m)

        ax2.plot(filtered_times, filtered_reserved, label=label)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Reserved Memory")
    ax2.set_title("Reserved Memory")
    ax2.legend()
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

    fig1, fig2 = compare_memory_usage(data, labels, args.memory_threshold)
    fig1.savefig(f"{DIR}/allocated_memory.png")
    fig2.savefig(f"{DIR}/reserved_memory.png")
