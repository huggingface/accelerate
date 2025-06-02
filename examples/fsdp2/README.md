## FSDP2 Examples

This folder contains examples of using FSDP2 with Accelerate, utilizing extra methods to improve training speed, performance or accuracy.

### FSDP2 + ao Float8Linear

In file `fsdp2_fp8.py` we use `ao Float8Linear` to train a model partially in FP8 precision. We utilize `AORecipeKwargs` to pass the `Float8LinearConfig` to the accelerator, 
which replaces the default `torch.nn.Linear` with `Float8Linear` from `ao`. We also utilize `TorchDynamoPlugin` together with regional compilation to compile the model,
gaining even more speed and memory savings, as `ao` doesn't ship with any kernels by default, so we have to gain the performance from compiling the model.

Replacing linear layers with `Float8Linear` can greatly improve performance, if used correctly and on hardware that supports FP8 tensor cores. This highly depends on the model dimensions and sequence length used for training.
You can view the performance of `Float8Linear` as a function of matrix dimensions in [this document](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md#performance). 

In our example, we use a 8B Llama3.1 model, which has a hidden dimension of 4096 and we train on sequence length of 8192. In the below images, we can see that this improves performance by ~25% compared to `bf16`, reaching ~10000 tokens per second, per device on 8x H100 GPUs, compared to ~8000 tokens per second using `bf16`, while loss function stays roughly the same.

<div style="display: flex; gap: 25px;">
  <div style="text-align: center; width: 49%;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_tps.png" alt="tps" style="width: 100%;">
    <p style="text-align: center; margin-top: 8px;">TPs per device, bf16 vs fp8</p>
  </div>
  <div style="text-align: center; width: 49%;">  
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_loss.png" alt="loss" style="width: 100%; max-width: 900px;">
    <p style="text-align: center; margin-top: 8px;">Loss curve, bf16 vs fp8</p>
  </div>
</div>

The figures above were generated on 8x H100 SXM GPUs, with 8192 sequence length and 1000 steps. If `--bf16` is specified, the model will be trained in bf16 precision, otherwise it will be trained in fp8 precision. To run the example, you can use the following command:

```bash
accelerate launch --fsdp2_fp8.py --sequence_length 8192 --num_steps 1000 --log_with wandb [--bf16]
```

