## FSDP2 Examples

This folder contains examples of using FSDP2 with Accelerate, utilizing extra methods to improve training speed, performance or accuracy.

### FSDP2 + TorchAO Float8Linear

In file `fsdp2_fp8.py` we use `TorchAO Float8Linear` to train a model partially in FP8 precision. We utilize `AORecipeKwargs` to pass the `Float8LinearConfig` to the accelerator, 
which replaces the default `torch.nn.Linear` with `Float8Linear` from `TorchAO`. We also utilize `TorchDynamoPlugin` together with regional compilation to compile the model,
gaining even more speed and memory savings.

Replacing linear layers with `Float8Linear` can greatly improve performance, if used correctly. This highly depends on the model dimensions and sequence length used for training.
You can view the performance of `Float8Linear` in [this document](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md#performance). In our example, we use a 8B Llama3.1
model, which has a hidden dimension of 4096 and we train on sequence length of 8192. In the below images, we can see that this improves performance by ~25% compared to `bf16`, reaching ~10000 tokens per second, per device on 8x H100 GPUs, compared to ~8000 tokens per second using `bf16`. 


<div style="text-align: center; margin-bottom: 32px;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_loss.png" alt="loss" style="width: 100%; max-width: 900px;">
  <p style="text-align: center; margin-top: 8px;">Loss curve, bf16 vs fp8</p>
</div>

<div style="display: flex; gap: 25px;">
  <div style="text-align: center; width: 49%;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_tps.png" alt="tps" style="width: 100%;">
    <p style="text-align: center; margin-top: 8px;">TPs per device, bf16 vs fp8</p>
  </div>
  <div style="text-align: center; width: 49%;">  
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/examples/fsdp2/fp8_steps.png" alt="steps" style="width: 100%;">
    <p style="text-align: center; margin-top: 8px;">Steps per second, bf16 vs fp8</p>
  </div>
</div>
