# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import DistributedType, set_seed


def step_model(model, input, target, accelerator, do_backward=True):
    model.train()
    output = model(input)
    loss = F.mse_loss(output, target.to(output.device))
    if not do_backward:
        loss /= accelerator.gradient_accumulation_steps
        loss.backward()
    else:
        accelerator.backward(loss)


def get_training_setup(accelerator, sched=False):
    "Returns everything needed to perform basic training"
    set_seed(42)
    model = RegressionModel()
    model.to(accelerator.device)
    dset = RegressionDataset()
    dataloader = DataLoader(dset, batch_size=16)
    ddp_model = deepcopy(model)
    if sched:
        opt = AdamW(params=model.parameters(), lr=1e-3)
        ddp_opt = AdamW(params=ddp_model.parameters(), lr=1e-3)
        sched = LambdaLR(opt, lr_lambda=lambda epoch: 0.65**epoch)
        ddp_sched = LambdaLR(ddp_opt, lr_lambda=lambda epoch: 0.65**epoch)
    # Make a copy of `model`
    if sched:
        ddp_model, ddp_opt, ddp_sched, dataloader = accelerator.prepare(ddp_model, ddp_opt, ddp_sched, dataloader)
    else:
        ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    # Use a single batch for all of the tests
    ddp_input, ddp_target = next(iter(dataloader)).values()
    if sched:
        return (
            model,
            opt,
            sched,
            ddp_model,
            ddp_opt,
            ddp_sched,
            ddp_input,
            ddp_target,
        )
    return model, ddp_model, ddp_input, ddp_target


def test_noop_sync(accelerator):
    # Test when on a single CPU or GPU that the context manager does nothing
    model, ddp_model, ddp_input, ddp_target = get_training_setup(accelerator)
    for iteration in range(3):
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator)
        # Do "gradient accumulation" (noop)
        if iteration % 2 == 0:
            # Accumulate grads locally
            with accelerator.no_sync(ddp_model):
                step_model(ddp_model, ddp_input, ddp_target, accelerator)
        else:
            # Sync grads
            step_model(ddp_model, ddp_input, ddp_target, accelerator)

        # Since `no_sync` is a noop, `ddp_model` and `model` grads should always be in sync
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            assert torch.allclose(
                param.grad, ddp_param.grad
            ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(16)]


def test_distributed_sync(accelerator):
    # Test on distributed setup that context manager behaves properly
    model, ddp_model, ddp_input, ddp_target = get_training_setup(accelerator)
    for iteration in range(3):
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator)
        # Do "gradient accumulation" (noop)
        if iteration % 2 == 0:
            # Accumulate grads locally
            with accelerator.no_sync(ddp_model):
                step_model(ddp_model, ddp_input, ddp_target, accelerator)
        else:
            # Sync grads
            step_model(ddp_model, ddp_input, ddp_target, accelerator)

        # DDP model and model should only be in sync when not (iteration % 2 == 0)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            if iteration % 2 == 0:
                # Grads should not be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is False
                ), f"Gradients in sync when they should not be:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})"
            else:
                # Grads should be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is True
                ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(16)]


def test_gradient_accumulation():
    accelerator = Accelerator(gradient_accumulation_steps=2)
    # Test that context manager behaves properly
    model, ddp_model, ddp_input, ddp_target = get_training_setup(accelerator)
    for iteration in range(3):
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator, False)
        # Do "gradient accumulation" (noop)
        accelerator.print(f'Should sync outside ctx: {accelerator.state.sync_gradients}')
        with accelerator.accumulate(ddp_model, [0, 1, 2]):
            accelerator.print(f'Should sync: {accelerator.state.sync_gradients}')
            step_model(ddp_model, ddp_input, ddp_target, accelerator)

        # DDP model and model should only be in sync when not (iteration % 2 == 0)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            if iteration % 2 == 0:
                # Grads should not be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is False
                ), f"Gradients in sync when they should not be:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})"
            else:
                # Grads should be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is True
                ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(16)]


def test_gradient_accumulation_with_opt_and_scheduler():
    accelerator = Accelerator(gradient_accumulation_steps=2)
    # Test that context manager behaves properly
    model, opt, sched, ddp_model, ddp_opt, ddp_sched, ddp_input, ddp_target = get_training_setup(accelerator, True)
    for iteration in range(3):
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator, False)
        opt.step()
        sched.step()
        opt.zero_grad()
        # Do "gradient accumulation" (noop)
        with accelerator.accumulate(ddp_model, [0, 1, 2]):
            step_model(ddp_model, ddp_input, ddp_target, accelerator)
            ddp_opt.step()
            ddp_sched.step()
            ddp_opt.zero_grad()

        # DDP model and model should only be in sync when not (iteration % 2 == 0)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            if iteration % 2 == 0:
                # Grads should not be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is False
                ), f"Gradients in sync when they should not be:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})"
            else:
                # Grads should be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is True
                ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"

        # DDP schedule and DDP optimizer should only be in sync when not (iteration % 2 == 0)
        if iteration % 2 == 0:
            # States should not be in sync
            assert opt.state != ddp_opt.state
            assert sched.last_epoch != ddp_sched.last_epoch
        else:
            # States should be in sync
            assert opt.state == ddp_opt.state
            assert sched.last_epoch == ddp_sched.last_epoch
        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(16)]


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.distributed_type == DistributedType.NO:
        if state.local_process_index == 0:
            print("**NOOP `no_sync` gradient accumulation**")
        test_noop_sync(accelerator)
    if state.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_CPU):
        if state.local_process_index == 0:
            print("**Distributed `no_sync` gradient accumulation**")
        test_distributed_sync(accelerator)
    print("**Test `accumulate` gradient accumulation**")
    test_gradient_accumulation()
    print("**Test `accumulate` gradient accumulation with optimizer and scheduler**")
    test_gradient_accumulation_with_opt_and_scheduler()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
