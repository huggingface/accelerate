# Compilation

## Overview

Pytorch 2.0 introduced `torch.compile`, a powerful feature that makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels. Key features of `torch.compile` include:

- **Performance Improvement**: Significantly speeds up model execution by optimizing the computation graph.
- **Ease of Use**: Requires minimal code changes to implement, making it highly accessible.
- **Compatibility**: Works seamlessly with existing PyTorch code and models.

When used with Accelerate, `torch.compile` integrates smoothly into distributed training workflows, allowing you to benefit from both distributed execution and compilation optimizations simultaneously.

The first execution of compiled code typically takes longer as it includes the compilation time, but subsequent runs are significantly faster. For optimal performance in different scenarios, `torch.compile` offers various modes like `"default"`, `"reduce-overhead"` (which uses CUDA graphs to further reduce overhead), and `"max-autotune"` (which performs extensive autotuning to find the best kernels for your model).

## Using `torch.compile` with Accelerate

Accelerate provides `TorchDynamoPlugin` for easy integration with `torch.compile`.

```python
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

# Configure the compilation backend
dynamo_plugin = TorchDynamoPlugin(
    backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
    mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
    fullgraph=True,
    dynamic=False
)

# Initialize accelerator with the plugin
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
# This will apply torch.compile to your model
model = accelerator.prepare(model)
```

## Regional Compilation

Instead of trying to compile the whole model, which usually has a big problem space for optimization. Regional compilation finds and compiles a repeated block first (usually a decoder/encoder layer), so that the compiler would re-use its cached/optimized generated code for the other blocks, reducing the cold start compilation time observed on the first inference call.

### How to Use Regional Compilation

To apply regional compilation, uou can simply enable it by setting `use_regional_compilation=True` in the `TorchDynamoPlugin` configuration. This will automatically apply regional compilation to your model.

```python
# Configure the compilation backend
dynamo_plugin = TorchDynamoPlugin(
    use_regional_compilation=True,
    ... # other parameters
)
# Initialize accelerator with the plugin
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
# This will apply compile_regions to your model
model = accelerator.prepare(model)
```

### Benefits of Regional Compilation

We have conducted extensive benchmarks comparing full compilation and regional compilation using the `torch.compile` feature in PyTorch. The full results are available in the [accelerate repository](https://github.com/huggingface/accelerate/tree/main/benchmarks/torch.compile/regional_compilation). The key findings from our benchmarks are:

1. **Comparable Performance**: Regional compilation delivers performance speedups similar to full compilation, especially for larger models.
2. **Faster Compilation**: Regional compilation significantly reduces the time taken to compile models, making it a more efficient choice for deployment.
3. **Batch Size Impact**: The performance difference between compilation strategies diminishes with larger batch sizes, indicating that the overhead of compilation is less impactful in those scenarios.
4. **Model Size Consideration**: The benefits of regional compilation are more pronounced in larger models, where the compilation time savings can be substantial.
5. **Practical Application**: For real-world applications, regional compilation is a practical choice for optimizing training cold start times, especially when working with large models.

## Conclusion

Both full and regional compilation can significantly speed up your models. Regional compilation offers a practical balance between compilation time and runtime performance, especially for training large models with substantial batch sizes.
