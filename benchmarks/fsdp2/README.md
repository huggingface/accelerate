# FSDP2 Benchmarks

This benchmark showcases `FSDP2` in 🤗 `accelerate` and compares it to `torch` baseline.

## Overview

This benchmark consists of two parts:
- `main.py` is the main script that runs the benchmark
- `visualize.py` is the script that visualizes the results (if `--output_dir` was specified for the previous command)

## Motivation

We want to showcase that 🤗 `accelerate`'s integration of `FSDP2` is on par raw PyTorch, and highlight a "broken" part in PyTorch that creating an optimizer before applying `FSDP2` **doesn't result in a working training loop**. (more on this later)
This script showcases **matching memory usage and convergence between `accelerate` and `torch`'s baseline.**
To deal with this breaking change (and maintain backward compatibility with FSDP1 in terms of an API), `accelerate` had to come up with a workaround since `accelerate` assumes that the user will nearly always create a model, optimizer, scheduler, etc beforehand and bring them themselves. This lead to an issue of a stark increase in memory as well as the model not even training if the user creates an optimizer beforehand. 
To workaround this, we replace the parameters inside the optimizer with the newly created FSDP2 sharded ones. More about this can be found in this [blog post (TBD)](TODO)
> [!WARNING]
> This script is intended to fit on 2x 24GB GPUs, though on so few GPUs it's not possible to see the memory difference (discrepancies in grad allocation result in lower memory usage in the non-fixed case), only the difference in convergence. Below are attached results from 8x H100 GPUs where the difference is visible.
> TLDR: more GPUs = bigger memory difference between fixed and non-fixed cases.

## Results

Here are the results from running the benchmark on 8x H100 GPUs:

<p align="center">
  <img src="imgs/allocated_memory.png" width="80%" alt="Allocated Memory Usage">
</p>
<p align="center">
  <img src="imgs/reserved_memory.png" width="80%" alt="Reserved Memory Usage">
</p>

As you can see, the memory usage of `accelerate` and `torch_post_shard` (the **intended** way) are very similar, while `torch_pre_shard_not_fixed` uses significantly more memory. Our fix in `torch_pre_shard_fixed` brings the memory usage back in line with the **intended** approach.

> [!WARNING]
> Timing discrepancies are due to the benchmarks being ran in 1 script.


## Running

To run the benchmark, you can either use `accelerate launch` or `torchrun`:
```bash
accelerate launch main.py
```
```bash
# For two GPUs
torchrun --nproc_per_node 2 main.py
```

This supports multiple configurable options, you can learn about them by running:
```bash
python3 main.py --help
```

This script will run 4 different benchmarks:
- `torch_optimizer_after_fsdp`: `torch` baseline where optimizer is created after applying `FSDP2`, this is the **intended** way to do it
- `torch_optimizer_before_fsdp_not_fixed`: `torch` baseline where optimizer is created before applying `FSDP2` without fixing the optimizer parameters
- `torch_optimizer_before_fsdp_fixed`: `torch` baseline where optimizer is created before applying `FSDP2` with our fix to the optimizer
- `accelerate`: `accelerate`'s own integration of `FSDP2` where optimizer is created before applying `FSDP2`, but we apply our fix to the optimizer

Memory results are saved in a folder specified by `--output_dir` argument.
Optionally, you can specify `--save_memory_snapshot` to save the torch memory snapshot, which can then be viewed using [`torch memory viz`](https://pytorch.org/memory_viz)

## Visualizing results

To visualize the results, you can run:

```bash
python3 visualize.py --dir <path_to_output_dir>
```

This will then create two plots, showcasing allocated and reserved memory usage between all the different benchmarks discussed above.



