# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import argparse
import functools
import itertools
import unittest
from typing import Any, Callable

import torch
from huggingface_hub import hf_hub_download
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import _recursive_wrap, transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.test_utils import execute_subprocess_async, get_torch_dist_unique_port, require_multi_gpu
from accelerate.test_utils.testing import require_torch_min_version, require_transformers
from accelerate.utils.imports import is_transformers_available


if is_transformers_available():
    from transformers import AutoConfig, AutoModel
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block


def manage_process_group(func: Callable[..., Any]) -> Callable[..., Any]:
    """Manage the creation and destruction of the distributed process group for the wrapped function."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        dist.init_process_group(world_size=torch.cuda.device_count())
        try:
            return func(*args, **kwargs)
        finally:
            dist.destroy_process_group()

    return wrapped


@manage_process_group
def load_checkpoint_and_dispatch_fsdp2():
    torch.cuda.set_device(device := torch.device(dist.get_rank()))

    pretrained_model_name_or_path = "bigscience/bloom-560m"
    model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

    model = AutoModel.from_pretrained(pretrained_model_name_or_path, device_map=device)
    assert isinstance(model, nn.Module)

    with init_empty_weights():
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        fsdp2_model = AutoModel.from_config(config)
        fsdp2_model.tie_weights()
        assert isinstance(fsdp2_model, nn.Module)

    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    fsdp2_model, _ = _recursive_wrap(
        fsdp2_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                GPT2Block,
                type(fsdp2_model),
            },
        ),
        wrapper_cls=functools.partial(
            fully_shard,
            mesh=mesh,
        ),
        ignored_modules=set(),
        ignored_params=set(),
    )

    fsdp2_model._apply(
        lambda t: torch.empty_like(t, device=device) if t.device == torch.device("meta") else t.to(device)
    )

    load_checkpoint_and_dispatch(fsdp2_model, model_path, strict=True, broadcast_from_rank0=True)

    for (name, tensor), (fsdp2_name, fsdp2_tensor) in zip(
        itertools.chain(model.named_parameters(), model.named_buffers()),
        itertools.chain(fsdp2_model.named_parameters(), fsdp2_model.named_buffers()),
    ):
        assert name == fsdp2_name
        assert isinstance(fsdp2_tensor, DTensor), fsdp2_name
        torch.testing.assert_close(tensor, fsdp2_tensor.full_tensor(), msg=fsdp2_name)


@manage_process_group
def load_checkpoint_and_dispatch_no_broadcast_from_rank0():
    torch.cuda.set_device(device := torch.device(dist.get_rank()))

    pretrained_model_name_or_path = "bigscience/bloom-560m"
    model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

    with init_empty_weights():
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        broadcasted_model = AutoModel.from_config(config)
        broadcasted_model.tie_weights()
        assert isinstance(broadcasted_model, nn.Module)

    broadcasted_model._apply(
        lambda t: torch.empty_like(t, device=device) if t.device == torch.device("meta") else t.to(device)
    )

    load_checkpoint_and_dispatch(broadcasted_model, model_path, strict=True, broadcast_from_rank0=True)

    with init_empty_weights():
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        non_broadcasted_model = AutoModel.from_config(config)
        non_broadcasted_model.tie_weights()
        assert isinstance(non_broadcasted_model, nn.Module)

    non_broadcasted_model._apply(
        lambda t: torch.empty_like(t, device=device) if t.device == torch.device("meta") else t.to(device)
    )

    load_checkpoint_and_dispatch(non_broadcasted_model, model_path, strict=True, broadcast_from_rank0=False)

    for (broadcasted_name, broadcasted_tensor), (non_broadcasted_name, non_broadcasted_tensor) in zip(
        itertools.chain(broadcasted_model.named_parameters(), broadcasted_model.named_buffers()),
        itertools.chain(non_broadcasted_model.named_parameters(), non_broadcasted_model.named_buffers()),
    ):
        assert broadcasted_name == non_broadcasted_name
        torch.testing.assert_close(broadcasted_tensor, non_broadcasted_tensor, msg=broadcasted_name)


@manage_process_group
def load_checkpoint_and_dispatch_ddp():
    torch.cuda.set_device(device := torch.device(dist.get_rank()))

    pretrained_model_name_or_path = "bigscience/bloom-560m"
    model_path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")

    model = AutoModel.from_pretrained(pretrained_model_name_or_path, device_map=device)
    assert isinstance(model, nn.Module)

    with init_empty_weights():
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        ddp_model = AutoModel.from_config(config)
        ddp_model.tie_weights()
        assert isinstance(ddp_model, nn.Module)

    ddp_model._apply(
        lambda t: torch.empty_like(t, device=device) if t.device == torch.device("meta") else t.to(device)
    )
    ddp_model = DistributedDataParallel(ddp_model)

    load_checkpoint_and_dispatch(ddp_model.module, model_path, strict=True, broadcast_from_rank0=True)

    for (name, tensor), (ddp_name, ddp_tensor) in zip(
        itertools.chain(model.named_parameters(), model.named_buffers()),
        itertools.chain(ddp_model.module.named_parameters(), ddp_model.module.named_buffers()),
    ):
        assert name == ddp_name
        torch.testing.assert_close(tensor, ddp_tensor, msg=ddp_name)


@require_torch_min_version(version="2.4.0")
@require_transformers
@require_multi_gpu
class TestLoadCheckpointAndDispatchWithBroadcast(unittest.TestCase):
    def test_load_checkpoint_and_dispatch_fsdp2(self):
        execute_subprocess_async(
            cmd=[
                "torchrun",
                f"--nproc_per_node={torch.cuda.device_count()}",
                f"--master_port={get_torch_dist_unique_port()}",
                __file__,
                "--fsdp2",
            ],
        )
        # successful return here == success - any errors would have caused an error in the sub-call

    def test_load_checkpoint_and_dispatch_no_broadcast_from_rank0(self):
        execute_subprocess_async(
            cmd=[
                "torchrun",
                f"--nproc_per_node={torch.cuda.device_count()}",
                f"--master_port={get_torch_dist_unique_port()}",
                __file__,
                "--no_broadcast_from_rank0",
            ],
        )
        # successful return here == success - any errors would have caused an error in the sub-call

    def test_load_checkpoint_and_dispatch_ddp(self):
        execute_subprocess_async(
            cmd=[
                "torchrun",
                f"--nproc_per_node={torch.cuda.device_count()}",
                f"--master_port={get_torch_dist_unique_port()}",
                __file__,
                "--ddp",
            ],
        )
        # successful return here == success - any errors would have caused an error in the sub-call


if __name__ == "__main__":
    # The script below is meant to be run under torch.distributed, on a machine with multiple GPUs:
    #
    # PYTHONPATH="src" python -m torch.distributed.run --nproc_per_node 2 --output_dir output_dir ./tests/test_fsdp2.py --fsdp2

    class CLIArgs(argparse.Namespace):
        fsdp2: bool
        ddp: bool
        no_broadcast_from_rank0: bool

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fsdp2", action="store_true")
    group.add_argument("--ddp", action="store_true")
    group.add_argument("--no_broadcast_from_rank0", action="store_true")
    args = parser.parse_args(namespace=CLIArgs())

    if args.fsdp2:
        load_checkpoint_and_dispatch_fsdp2()
    elif args.ddp:
        load_checkpoint_and_dispatch_ddp
    elif args.no_broadcast_from_rank0:
        load_checkpoint_and_dispatch_no_broadcast_from_rank0()
    else:
        raise ValueError("Missing test selection")
