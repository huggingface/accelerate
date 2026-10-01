# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Measure CUDA AMP optimizer overhead, synchronizing only at timing boundaries.

Run against a checkout with ``PYTHONPATH=src python benchmarks/amp/benchmark_optimizer.py``.
Pass ``--optimizer-source LABEL PATH`` once per exported optimizer.py to compare
revisions with randomly interleaved timing blocks on the same workload.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

import torch

from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler


def benchmark(optimizer_class, *, width, fused, scheduler_enabled, iterations, warmup, repeats):
    torch.manual_seed(42)
    model = torch.nn.Linear(width, width, bias=False, device="cuda")
    inputs = torch.randn(32, width, device="cuda")
    raw_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=fused)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, growth_interval=1_000_000_000)
    optimizer = optimizer_class(raw_optimizer, scaler=scaler)
    scheduler = AcceleratedScheduler(
        torch.optim.lr_scheduler.LambdaLR(raw_optimizer, lambda _: 1.0), optimizer, split_batches=True
    )

    def step():
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(inputs).float().square().mean()
        scaler.scale(loss).backward()
        optimizer.step()
        if scheduler_enabled:
            scheduler.step()

    for _ in range(warmup):
        step()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)

    # Profile a separate iteration so instrumentation cannot affect the timings.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        step()
    scalar_reads = sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")
    return {
        "width": width,
        "fused": fused,
        "scheduler": scheduler_enabled,
        "milliseconds_per_step": samples,
        "median_milliseconds_per_step": statistics.median(samples),
        "host_scalar_reads_per_step": scalar_reads,
        "final_scale": scaler.get_scale(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-source", nargs=2, action="append", metavar=("LABEL", "PATH"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--widths", nargs="+", type=int, default=[32, 1024])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    Accelerator()
    optimizer_classes = {"checkout": AcceleratedOptimizer}
    source_hashes = {}
    if args.optimizer_source:
        optimizer_classes = {}
    for label, filename in args.optimizer_source or []:
        if label in optimizer_classes:
            parser.error(f"Duplicate optimizer label: {label}")
        source = Path(filename)
        spec = importlib.util.spec_from_file_location(f"accelerate._amp_benchmark_optimizer_{label}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        optimizer_classes[label] = module.AcceleratedOptimizer
        source_hashes[label] = hashlib.sha256(source.read_bytes()).hexdigest()
    cases = [
        (width, fused, scheduler) for width in args.widths for fused in (False, True) for scheduler in (False, True)
    ]
    order = random.Random(42)
    order.shuffle(cases)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "optimizer_source_sha256": source_hashes,
        "blocks": [],
    }
    for width, fused, scheduler in cases:
        for repeat in range(args.repeats):
            labels = list(optimizer_classes)
            order.shuffle(labels)
            for label in labels:
                # Each block starts from the same seeded model and optimizer state.
                result = benchmark(
                    optimizer_classes[label],
                    width=width,
                    fused=fused,
                    scheduler_enabled=scheduler,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    repeats=1,
                )
                result.update(label=label, repeat=repeat)
                report["blocks"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
