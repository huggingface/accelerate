# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Compare the next training update after a fresh-process DeepSpeed checkpoint load."""

import argparse
from pathlib import Path

import deepspeed
import torch
from deepspeed.accelerator import get_accelerator

from accelerate import Accelerator, PartialState
from accelerate.utils import DeepSpeedPlugin, DistributedType, set_seed
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler


def main(checkpoint_dir, resume):
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2},
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "Muon",
            "params": {"lr": 1e-5, "muon_lr": 0.02, "adam_lr": 1e-5, "torch_adam": True},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "warmup_type": "linear",
                "total_num_steps": 12,
            },
        },
    }
    cpu = get_accelerator().device_name() == "cpu"
    if cpu:
        # Use real Gloo collectives without requiring the optional shared-memory extension.
        deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    plugin = DeepSpeedPlugin(hf_ds_config=config, zero3_init_flag=False)
    accelerator = Accelerator(cpu=cpu, mixed_precision="no", deepspeed_plugin=plugin)
    if cpu:
        # Accelerator(cpu=True) selects MULTI_CPU; exercise the DeepSpeed integration with a real CPU engine.
        PartialState().distributed_type = DistributedType.DEEPSPEED
        accelerator.state.distributed_type = DistributedType.DEEPSPEED
        accelerator.state.deepspeed_plugins = plugin
        accelerator.deepspeed_engine_wrapped = None

    set_seed(4367)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 8))
    optimizer = DummyOptim(model.parameters(), lr=1e-5)
    scheduler = DummyScheduler(optimizer, warmup_num_steps=4)
    engine, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    assert engine.lr_scheduler is scheduler.scheduler
    assert engine.lr_scheduler.optimizer is engine.optimizer.optimizer
    torch.testing.assert_close(
        torch.tensor(engine.lr_scheduler.max_lrs), torch.tensor([0.02, 1e-5]), rtol=1e-6, atol=0
    )

    inputs = torch.sin(torch.arange(64, device=accelerator.device).float()).reshape(4, 16)

    def train_step(index):
        # Changing gradient directions makes the next update sensitive to restored Adam moments.
        targets = torch.full((4, 8), (-1.0) ** index, device=accelerator.device)
        loss = (engine(inputs) - targets).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    checkpoint_path = checkpoint_dir / "checkpoint"
    reference_path = checkpoint_dir / "next_step.pt"
    if resume:
        accelerator.load_state(checkpoint_path)
    else:
        for index in range(3):
            train_step(index)
        accelerator.save_state(checkpoint_path)

    assert engine.global_steps == 3
    assert engine.lr_scheduler.last_batch_iteration == 2
    # These are the configured peaks at halfway through a four-step linear warmup.
    torch.testing.assert_close(torch.tensor(engine.get_lr()), torch.tensor([0.01, 5e-6]), rtol=1e-6, atol=0)
    before_update = {name: param.detach().cpu().clone() for name, param in engine.module.named_parameters()}
    train_step(3)
    assert engine.global_steps == 4
    torch.testing.assert_close(torch.tensor(engine.get_lr()), torch.tensor([0.015, 7.5e-6]), rtol=1e-6, atol=0)
    parameters = {name: param.detach().cpu() for name, param in engine.module.named_parameters()}
    deltas = {name: param - before_update[name] for name, param in parameters.items()}
    assert any(not torch.equal(before_update[name], param) for name, param in parameters.items())
    if resume:
        reference = torch.load(reference_path, map_location="cpu")
        torch.testing.assert_close(parameters, reference["parameters"], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(deltas, reference["deltas"], rtol=1e-6, atol=1e-8)
        assert engine.get_lr() == reference["lrs"]
    else:
        torch.save({"parameters": parameters, "deltas": deltas, "lrs": engine.get_lr()}, reference_path)

    torch.distributed.barrier()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.checkpoint_dir, args.resume)
