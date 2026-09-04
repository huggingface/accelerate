<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Sharding iterable datasets

A map-style dataset is sharded by its sampler: [`~Accelerator.prepare`] replaces the `batch_sampler` so each process
draws different indices, and no process touches data that belongs to another one.

An `IterableDataset` has no indices to hand out, so it is sharded differently, and the difference has consequences that
are easy to miss. This guide explains what [`~Accelerator.prepare`] does to an iterable dataset, when to shard the
dataset yourself instead, and what you have to guarantee if you do.

## What `prepare` does to an iterable dataset

When you pass a dataloader over a `torch.utils.data.IterableDataset` to [`~Accelerator.prepare`], the dataset is
wrapped in an [`~data_loader.IterableDatasetShard`]:

```python
accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)
# dataloader.dataset is now an IterableDatasetShard around your dataset
```

The wrapper shards *elements*, not sources. On every process it iterates **the entire underlying dataset**, buffers
`batch_size * num_processes` elements, and yields only the `batch_size` elements that belong to its own process index.

This is correct — the processes see disjoint elements and yield the same number of batches — but every process reads
everything. With 8 shards of 10 records, `num_processes=4` and `batch_size=2`:

| | records per process | unique records seen | shard reads |
|---|---|---|---|
| your dataset alone | — | 80 | 8 |
| wrapped by `prepare` | 20, 20, 20, 20 | 80 | **32** (4×) |

If reading is cheap (an in-memory generator, a small local file) this costs nothing worth thinking about. If each
element comes off remote storage, or is decoded, or the "read" is a query, you are paying for it `num_processes` times.

<Tip warning={true}>

Dataloader workers multiply this again. PyTorch replicates an `IterableDataset` into every worker process, so with
`num_workers=2` each process reads everything twice and yields every element twice — `num_processes × num_workers`
reads in total. This is the standard `IterableDataset` caveat and the wrapper does not fix it: splitting work across
workers with `torch.utils.data.get_worker_info()` is still your responsibility.

</Tip>

<Tip warning={true}>

The wrapper assumes every process iterates the underlying dataset in the **same order**. If your dataset shuffles
itself with a process-local seed, the per-process slices no longer partition the data and processes both overlap and
drop elements. Use a synchronized `generator` attribute for any randomization inside the dataset, as described in
[Accelerate's internal mechanism](internal_mechanism).

</Tip>

## Datasets that already shard themselves

Streaming pipelines often shard by source: each process opens its own files, its own shards, its own byte ranges, so
nothing is read twice. If you pass such a dataloader to [`~Accelerator.prepare`], the wrapper is applied **on top** of
the sharding you already did, and shards the stream a second time.

Nothing errors and nothing hangs, but each process keeps roughly `1 / num_processes` of the data it was given. Same
setup as above (8 shards of 10 records, `num_processes=4`, `batch_size=2`), with a dataset that splits the shard list
by process index:

| | records per process | unique records seen |
|---|---|---|
| without `prepare` | 20, 20, 20, 20 | 80 / 80 |
| with `prepare` | 6, 6, 6, 6 | **24 / 80** |

An epoch that silently covers a third of the data looks exactly like a normal epoch in the loss curve. The visible
symptom is the step count: `len()` of the wrapper is `ceil(len(dataset) / (batch_size * num_processes)) * batch_size`,
so if your dataset defines `__len__`, the reported length drops by a factor of `num_processes` as well.

## Choosing an approach

| your dataset | what to do |
|---|---|
| `datasets.IterableDataset` with `n_shards >= num_processes` | call [`~Accelerator.prepare`]; Accelerate calls `.shard()` on it and each process reads only its shards |
| plain `IterableDataset`, cheap to read | call [`~Accelerator.prepare`]; accept that each process reads everything |
| plain `IterableDataset` that shards by source itself | do **not** pass the dataloader to [`~Accelerator.prepare`] (see below) |
| one stream that cannot be split, expensive to decode | `DataLoaderConfiguration(dispatch_batches=True)`: process 0 reads and scatters batches to the others |

<Tip>

`dispatch_batches=True` is the option to reach for when the data cannot be split at the source but you still want it
read once. It makes process 0 the only reader, which is a bottleneck if reading is the slow part — that is the
trade-off against sharding by source.

</Tip>

## Sharding by source yourself

Resolve the process identity **when the dataset is built**, in the main process, and split by worker inside
`__iter__`:

```python
from accelerate import PartialState
from torch.utils.data import IterableDataset, get_worker_info


class ShardedDataset(IterableDataset):
    def __init__(self, files):
        # sorted, so every process derives the same ordering and never reads another one's files
        self.files = sorted(files)
        state = PartialState()
        self.num_processes, self.process_index = state.num_processes, state.process_index

    def __iter__(self):
        worker = get_worker_info()
        num_workers, worker_id = (1, 0) if worker is None else (worker.num_workers, worker.id)
        for file in self.files[self.process_index :: self.num_processes][worker_id::num_workers]:
            yield from read(file)
```

Then prepare everything except the dataloader, and move the batches yourself:

```python
model, optimizer = accelerator.prepare(model, optimizer)

for batch in dataloader:
    batch = batch.to(accelerator.device)
    ...
```

<Tip warning={true}>

Build the dataset **after** the [`Accelerator`] (or [`PartialState`]) exists. Resolving the process identity earlier
returns `num_processes=1`, every process keeps the whole file list, and the sharding becomes a silent no-op.

Do not construct a [`PartialState`] inside `__iter__` either: with `num_workers > 0` that code runs in a dataloader
worker, where it may try to set up the distributed environment again.

</Tip>

### Every process must yield the same number of batches

[`~data_loader.IterableDatasetShard`] guarantees this for you — it only yields from complete
`batch_size * num_processes` windows, and with `drop_last=False` it fills the last incomplete window by cycling back to
the elements it started with (in the measurement above, 9 elements over a window of 8 are yielded as 16, with 7
repeats).

Once you shard the dataset yourself, that guarantee is yours. Gradients are synchronized with an all-reduce on every
step, so a process that runs out of data early leaves the loop and never makes its call, and the remaining processes
wait on it forever — with no traceback, no error, and no output. Files rarely hold the same number of records, so this
has to be handled explicitly, by truncating every process to the same count, padding the short ones, or wrapping the
loop in `DistributedDataParallel.join()` for uneven inputs.

### What you give up by not preparing the dataloader

- **Device placement.** Move batches with `batch.to(accelerator.device)`.
- **The end-of-epoch gradient sync.** [`~Accelerator.accumulate`] force-syncs on the last batch of an epoch by reading
  `end_of_dataloader` off the prepared dataloader. Without one it always reads `False`, so accumulation is driven by
  the step counter alone and a final partial accumulation window is not synced. Choose a number of batches per epoch
  that is a multiple of `gradient_accumulation_steps` if that matters to you.
- **RNG synchronization** across processes at each new iteration.

## Checklist

- [ ] The dataset is built after the [`Accelerator`] / [`PartialState`], and `num_processes` is what you expect.
- [ ] The file list is in the same order on every process.
- [ ] Either the dataset shards by source and the dataloader is not prepared, or the dataloader is prepared and the
      dataset does not shard — never both.
- [ ] Work is split across dataloader workers as well as processes.
- [ ] Every process yields the same number of batches.
- [ ] Steps per epoch match the number of records you expect, divided by `num_processes`.
