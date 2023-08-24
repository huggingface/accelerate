<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Understanding how big of a model can fit on your machine

One very difficult aspect when exploring potential models to use on your machine is knowing just how big of a model will *fit* with your current graphics card.

To help alleviate this, 🤗 Accelerate has a CLI interface through `accelerate calculate-memory`. This tutorial will 
help walk you through using it, what to expect, and at the end link to the interactive demo hosted on the 🤗 Hub which will 
even let you post those results directly on the model repo!

Currently we support searching for models that can be used in `timm` and `transformers`.

<Tip>

    This API will load the model into memory on the `meta` device, so we are not actually downloading 
    and loading the full weights of the model into memory, nor do we need to. As a result it's 
    perfectly fine to measure 8 billion parameter models (or more), without having to worry about 
    if your CPU can handle it!

</Tip>

## The Command

When using `accelerate estimate-memory`, you need to pass in the name of the model you want to use, potentially the framework
that model utilizing (if it can't be found automatically), and the data types you want the model to be loaded in with.

For example, here is how we can calculate the memory footprint for `bert-base-cased`:

```bash
accelerate estimate-memory bert-base-cased
```

This will download the `config.json` for `bert-based-cased`, load the model on the `meta` device, and report back how much space
it will use:

```
┌────────────────────────────────────────────────────┐
│    Memory Usage for loading `bert-base-cased`      │
├───────┬─────────────┬──────────┬───────────────────┤
│ dtype │Largest Layer│Total Size│Training using Adam│
├───────┼─────────────┼──────────┼───────────────────┤
│float32│   84.95 MB  │413.18 MB │      1.61 GB      │
│float16│   42.47 MB  │206.59 MB │     826.36 MB     │
│  int8 │   21.24 MB  │103.29 MB │     413.18 MB     │
│  int4 │   10.62 MB  │ 51.65 MB │     206.59 MB     │
└───────┴─────────────┴──────────┴───────────────────┘
```

By default it will return all the supported dtypes (`int4` through `float32`), but if you are interested in specific ones these can be filtered.

### Specific libraries

If the source library cannot be determined automatically (like it could in the case of `bert-base-cased`), a library name can
be passed in. 

```bash
accelerate estimate-memory HuggingFaceM4/idefics-80b-instruct --library_name transformers
```

```
┌────────────────────────────────────────────────────────────────────┐
│   Memory Usage for loading `HuggingFaceM4/idefics-80b-instruct`    |
├───────┬─────────────┬──────────┬───────────────────────────────────┤
│ dtype │Largest Layer│Total Size│        Training using Adam        │
├───────┼─────────────┼──────────┼───────────────────────────────────┤
│float32│   3.02 GB   │297.12 GB │              1.16 TB              │
│float16│   1.51 GB   │148.56 GB │             594.24 GB             │
│  int8 │  772.52 MB  │ 74.28 GB │             297.12 GB             │
│  int4 │  386.26 MB  │ 37.14 GB │             148.56 GB             │
└───────┴─────────────┴──────────┴───────────────────────────────────┘
```

```bash
accelerate estimate-memory timm/resnet50.a1_in1k --library_name timm
```

```
┌────────────────────────────────────────────────────┐
│ Memory Usage for loading `timm/resnet50.a1_in1k`   │
├───────┬─────────────┬──────────┬───────────────────┤
│ dtype │Largest Layer│Total Size│Training using Adam│
├───────┼─────────────┼──────────┼───────────────────┤
│float32│    9.0 MB   │ 97.7 MB  │     390.78 MB     │
│float16│    4.5 MB   │ 48.85 MB │     195.39 MB     │
│  int8 │   2.25 MB   │ 24.42 MB │      97.7 MB      │
│  int4 │   1.12 MB   │ 12.21 MB │      48.85 MB     │
└───────┴─────────────┴──────────┴───────────────────┘
```

### Specific dtypes

As mentioned earlier, while we return `int4` through `float32` by default, any dtype can be used from `float32`, `float16`, `int8`, and `int4`.

To do so, pass them in after specifying `--dtypes`:

```bash
accelerate estimate-memory bert-base-cased --dtypes float32 float16
```

```
┌────────────────────────────────────────────────────┐
│    Memory Usage for loading `bert-base-cased`      │
├───────┬─────────────┬──────────┬───────────────────┤
│ dtype │Largest Layer│Total Size│Training using Adam│
├───────┼─────────────┼──────────┼───────────────────┤
│float32│   84.95 MB  │413.18 MB │      1.61 GB      │
│float16│   42.47 MB  │206.59 MB │     826.36 MB     │
└───────┴─────────────┴──────────┴───────────────────┘
```

## Caviats with this calculator

This calculator will tell you how much memory is needed to purely load the model in, *not* to perform inference.

In general, you can expect to add up to an additional 20% to this number as found by [EleutherAI](https://blog.eleuther.ai/transformer-math/). We'll be conducting research into finding a more accurate estimate to these values, and will update 
this calculator once done.

## Live Gradio Demo

Lastly, we invite you to try the [live Gradio demo](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) of this utility,
which includes an option to post a discussion thread on a models repository with this data. Doing so will help provide access to these numbers in the community faster and help users know what you've learned!