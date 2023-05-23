from .testing import (
    are_the_same_tensors,
    execute_subprocess_async,
    require_cpu,
    require_cuda,
    require_huggingface_suite,
    require_mps,
    require_multi_gpu,
    require_multi_xpu,
    require_safetensors,
    require_single_gpu,
    require_single_xpu,
    require_torch_min_version,
    require_tpu,
    require_xpu,
    skip,
    slow,
)
from .training import RegressionDataset, RegressionModel


from .scripts import test_script, test_sync, test_ops  # isort: skip
