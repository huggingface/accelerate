## FSDP2 Examples

This folder contains examples of using FSDP2 with Accelerate, utilizing extra methods to improve training speed, performance or accuracy.

### FSDP2 + ND Parallelism

With `ParallelismConfig`, you can use 🤗 accelerate to train models with n-dimensional parallelism. This builds on top of 🤗 transformers, which we utilize for tensor parallelism sharding.
Accelerate then takes care of everything else, such as data parallelism, FSDP, and more to come.
Script `nd_parallel.py` showcases just how you can do it. We enable you to configure 3 different parallel dimensions (for now 👀):
- dp_replicate_size: how many replicas of the model to create, each replica is trained on a different subset of the data and averaged at the end of each step, same as DDP in Torch
- dp_shard_size: across how many devices is the model sharded, this is utilizing FSDP2 to shard the model across devices, so each device has a different part of the model
- tp_size: how many devices to use for tensor parallelism, this is utilizing the tensor parallelism from 🤗 transformers

For example, with 8 nodes, you can run the script as such:
```bash
accelerate launch --num-processes 8 nd_parallel.py \
    --dp-replicate-size 2 \
    --dp-shard-size 2 \
    --tp-size 2 \
```

<Tip>
  Only use TP intra-node - therefore max TP size you should need is 8, you can also lower this as FSDP (`--dp-shard-size`) can be faster on smaller models with
  shorter sequence lengths. If you still cannot fit into memory, utilize `--dp-shard-size` as much as you can. Then to scale up to utilize all your GPUs, fill the rest
  with `--dp-replicate-size`. This is only a general guideline, you can (and should) experiment with different parallelism configurations to find the best one for your model and hardware. You can learn more about the general strategies for parallelism in our [blog](TODO) or if you wanna dive deep, read the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
</Tip>

We plan to add more parallelisms in the future, with context parallelism coming soon and pipeline parallelism being planned.

### FSDP2 + ao Float8Linear

In file `fsdp2_fp8.py` we use `Float8Linear` from `ao` to train a model partially in FP8 precision. We utilize `AORecipeKwargs` to pass the `Float8LinearConfig` to the accelerator, 
which replaces the default `torch.nn.Linear` with `Float8Linear`. We also utilize `TorchDynamoPlugin` together with regional compilation to compile the model,
gaining even more speed and memory savings, as `ao` doesn't ship with any kernels by default, so we have to gain the performance from compiling the model.

Replacing linear layers with `Float8Linear` can greatly improve performance, if used correctly and on hardware that supports FP8 tensor cores. This highly depends on the model dimensions and sequence length used for training.
You can view the performance of `Float8Linear` as a function of matrix dimensions in [this document](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md#performance). 

In our example, we use a 8B Llama3.1 model, which has a hidden dimension of 4096 and we train on sequence length of 8192. In the below images, we can see that this improves performance by ~25% compared to `bf16`, reaching ~10000 tokens per second, per device on 8x H100 GPUs, compared to ~8000 tokens per second using `bf16`, while loss function stays roughly the same. We can also see that the FLOPS raise by using FP8.

<div style="display: flex; gap: 25px;">
  <div style="text-align: center; width: 49%;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_tps.png" alt="tps" style="width: 100%;">
    <p style="text-align: center; margin-top: 8px;">TPs per device, bf16 vs fp8</p>
  </div>
  <div style="text-align: center; width: 49%;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_tflops.png" alt="tflops" style="width: 100%;">
    <p style="text-align: center; margin-top: 8px;">TFLOPS per device, bf16 vs fp8. We cannot really compare MFU as fp8 tensor cores are used as well.</p>
  </div>
  
  <div style="text-align: center; width: 49%;">  
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_loss.png" alt="loss" style="width: 100%; max-width: 900px;">
    <p style="text-align: center; margin-top: 8px;">Loss curve, bf16 vs fp8, it's hard to see the difference as the curves mostly overlap</p>
  </div>
</div>

The figures above were generated on 8x H100 SXM GPUs, with 8192 sequence length and 1000 steps. To run the example, you can use the following command, where you can specify the precision to train in:

```bash
accelerate launch fsdp2_fp8.py --sequence-length 8192 --num-steps 1000 --log_with wandb --precision [fp8 | bf16]
```

