# Deepspeed's ALST/Ulysses sequence parallelism

This is an example of the use of Ulysses Sequence Parallelism, which uses attention head parallelism and is part of the Arctic Long Sequence Training project at [ArcticTraining](https://github.com/snowflakedb/ArcticTraining). [This paper](https://arxiv.org/abs/2506.13996) goes into the details of this protocol.

For nuances of usage please refer to the main HF Accelerate tutorial on [Context Parallelism](https://huggingface.co/docs/accelerate/en/concept_guides/context_parallelism).

You need to use at least `2` gpus to enable ALST/Ulysses sequence parallelism.

To run the example with `4` gpus:

```bash
bash ./sp-alst.sh
```

Change `4` to the desired sequence parallelism degree in these 2 files:
```
sp-alst.accelerate-config.yml:num_processes: 4
sp-alst.py:    sp_size=4,
```
