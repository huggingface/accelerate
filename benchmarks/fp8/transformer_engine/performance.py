# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Measure single-GPU BF16/FP8 training performance and report MRPC quality separately."""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path


CASES = ("bf16", "te_bf16", "te_fp8", "accelerate_fp8")


def run_case(args):
    import evaluate
    import torch
    import transformer_engine.pytorch as te
    from fp8_utils import evaluate_model, get_named_parameters, get_training_utilities
    from transformer_engine.common.recipe import DelayedScaling, Format

    from accelerate import Accelerator
    from accelerate.utils import TERecipeKwargs, set_seed
    from accelerate.utils.transformer_engine import convert_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU to this benchmark.")
    if "fp8" in args.case and not te.is_fp8_available():
        raise RuntimeError(te.is_fp8_available(return_reason=True)[1])

    set_seed(42)
    recipe_kwargs = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    accelerator = Accelerator(
        mixed_precision="fp8" if args.case == "accelerate_fp8" else "bf16",
        kwargs_handlers=[TERecipeKwargs(**recipe_kwargs)] if args.case == "accelerate_fp8" else [],
    )
    model, optimizer, train_loader, eval_loader, scheduler = get_training_utilities(
        args.model_name, batch_size=args.batch_size, accelerator=accelerator
    )
    if args.case in ("te_bf16", "te_fp8"):
        old_params = get_named_parameters(model)
        with torch.no_grad():
            convert_model(model)
        new_params = get_named_parameters(model)
        mapping = {param: new_params[name] for name, param in old_params.items()}
        for group in optimizer.param_groups:
            group["params"] = [mapping[param] for param in group["params"]]
    if args.case == "accelerate_fp8":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model.to(accelerator.device)

    metric = evaluate.load("glue", "mrpc")
    before = evaluate_model(model, eval_loader, metric)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max")
    # Reset after initialization so dropout and shuffled batches start from the same seed in every case.
    set_seed(42)
    model.train()
    batches = iter(train_loader)
    samples = tokens = 0
    losses = []
    for step in range(args.warmup_steps + args.steps):
        if step == args.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        fp8_context = te.autocast(recipe=recipe) if args.case == "te_fp8" else nullcontext()
        with fp8_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        if args.case == "accelerate_fp8":
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach())
        if step >= args.warmup_steps:
            samples += batch["labels"].numel()
            tokens += batch["input_ids"].numel()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if not torch.isfinite(torch.stack(losses)).all():
        raise RuntimeError(f"Non-finite training loss in {args.case}.")
    after = evaluate_model(model, eval_loader, metric)
    result = {
        "case": args.case,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "samples": samples,
        "padded_tokens": tokens,
        "seconds": elapsed,
        "step_ms": 1000 * elapsed / args.steps,
        "samples_per_second": samples / elapsed,
        "padded_tokens_per_second": tokens / elapsed,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "initial_metrics": before,
        "trained_metrics": after,
        "first_loss": losses[0].item(),
        "last_loss": losses[-1].item(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformer_engine": version("transformer-engine"),
            "transformers": version("transformers"),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.case}: {result['samples_per_second']:.1f} samples/s, {result['step_ms']:.1f} ms/step", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.repeats) < 1 or args.warmup_steps < 1:
        parser.error("Batch size, steps, repeats, and warmup steps must be positive.")
    if args.case:
        if args.output is None:
            parser.error("A worker case requires --output.")
        run_case(args)
        return

    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for case in CASES:
            runs = []
            for repeat in range(args.repeats):
                output = Path(directory) / f"{case}-{repeat}.json"
                # Fresh processes isolate CUDA allocations, optimizer state, and TE caches between cases.
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--case",
                        case,
                        "--model-name",
                        args.model_name,
                        "--batch-size",
                        str(args.batch_size),
                        "--warmup-steps",
                        str(args.warmup_steps),
                        "--steps",
                        str(args.steps),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                runs.append(json.loads(output.read_text()))
            results[case] = {
                "median": {
                    key: statistics.median(run[key] for run in runs)
                    for key in (
                        "step_ms",
                        "samples_per_second",
                        "padded_tokens_per_second",
                        "peak_allocated_mib",
                        "peak_reserved_mib",
                    )
                },
                "runs": runs,
            }
    for case, result in results.items():
        for run, reference in zip(result["runs"], results["bf16"]["runs"]):
            if (run["samples"], run["padded_tokens"]) != (reference["samples"], reference["padded_tokens"]):
                raise RuntimeError(f"Measured workloads differ between {case} and BF16.")
        result["speedup_vs_bf16"] = results["bf16"]["median"]["step_ms"] / result["median"]["step_ms"]
        result["metric_deltas_vs_bf16"] = {
            metric: [
                run["trained_metrics"][metric] - reference["trained_metrics"][metric]
                for run, reference in zip(result["runs"], results["bf16"]["runs"])
            ]
            for metric in ("accuracy", "f1")
        }
    results["te_fp8"]["speedup_vs_te_bf16"] = (
        results["te_bf16"]["median"]["step_ms"] / results["te_fp8"]["median"]["step_ms"]
    )
    serialized = json.dumps(results, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
