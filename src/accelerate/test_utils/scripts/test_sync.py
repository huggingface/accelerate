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

from accelerate.accelerator import Accelerator, DataLoaderConfiguration, GradientAccumulationPlugin
from accelerate.state import GradientState
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import DistributedType, set_seed


def check_model_parameters(model_a, model_b, did_step, iteration, **kwargs):
    for param, grad_param in zip(model_a.parameters(), model_b.parameters()):
        if not param.requires_grad:
            continue
        if not did_step:
            # Grads should not be in sync
            assert (
                torch.allclose(param.grad, grad_param.grad, **kwargs) is False
            ), f"Gradients in sync when they should not be at iteration {iteration}:\nmodel_a grad ({param.grad}) == model_b grad ({grad_param.grad})"
        else:
            # Grads should be in sync
            assert (
                torch.allclose(param.grad, grad_param.grad, **kwargs) is True
            ), f"Gradients not in sync when they should be at iteration {iteration}:\nmodel_a grad ({param.grad}) != model_b grad ({grad_param.grad})"


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
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=80)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    if sched:
        opt = AdamW(params=model.parameters(), lr=1e-3)
        ddp_opt = AdamW(params=ddp_model.parameters(), lr=1e-3)
        sched = LambdaLR(opt, lr_lambda=lambda epoch: epoch**0.65)
        ddp_sched = LambdaLR(ddp_opt, lr_lambda=lambda epoch: epoch**0.65)
    # Make a copy of `model`
    if sched:
        ddp_model, ddp_opt, ddp_sched, dataloader = accelerator.prepare(ddp_model, ddp_opt, ddp_sched, dataloader)
    else:
        ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    if sched:
        return (model, opt, sched, dataloader, ddp_model, ddp_opt, ddp_sched)
    return model, ddp_model, dataloader


def test_noop_sync(accelerator):
    # Test when on a single CPU or GPU that the context manager does nothing
    model, ddp_model, dataloader = get_training_setup(accelerator)
    # Use a single batch
    ddp_input, ddp_target = next(iter(dataloader)).values()
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
        check_model_parameters(model, ddp_model, True, iteration)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            assert torch.allclose(
                param.grad, ddp_param.grad
            ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(len(ddp_input))]


def test_distributed_sync(accelerator):
    # Test on distributed setup that context manager behaves properly
    model, ddp_model, dataloader = get_training_setup(accelerator)
    # Use a single batch
    ddp_input, ddp_target = next(iter(dataloader)).values()
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
        ddp_input = ddp_input[torch.randperm(len(ddp_input))]


def test_distributed_sync_multiple_fwd(accelerator):
    # Test on distributed setup that context manager behaves properly when used with multiple forwards followed by multiple backwards
    model, ddp_model, dataloader = get_training_setup(accelerator)
    # Do multiple forwards
    losses = []
    num_iterations = 3
    for iteration in range(num_iterations):
        ddp_input, ddp_target = next(iter(dataloader)).values()

        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)

        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator)

        # Accumulate grads locally
        with accelerator.no_sync(ddp_model):
            ddp_output = ddp_model(ddp_input)
            loss = F.mse_loss(ddp_output, ddp_target.to(ddp_output.device))
            losses.append(loss)

    # Do multiple backwards and sync only at the last backward
    for iteration in range(num_iterations):
        loss = losses[iteration]

        if iteration < num_iterations - 1:
            # Accumulate grads locally
            accelerator.backward(loss)

            # DDP model and model should only be in sync after last backward
            for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
                if not param.requires_grad:
                    continue
                # Grads should not be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is False
                ), f"Gradients in sync when they should not be:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})"

        else:
            # Sync grads if last backward
            with accelerator.trigger_sync_in_backward(ddp_model):
                accelerator.backward(loss)

            # DDP model and model should only be in sync after last backward
            for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
                if not param.requires_grad:
                    continue
                # Grads should be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is True
                ), f"Gradients not in sync when they should be:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"


def test_gradient_accumulation(split_batches=False, dispatch_batches=False, sync_each_batch=False):
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2, sync_each_batch=sync_each_batch)
    accelerator = Accelerator(
        split_batches=split_batches,
        dispatch_batches=dispatch_batches,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    # Test that context manager behaves properly
    model, ddp_model, dataloader = get_training_setup(accelerator)
    for iteration, batch in enumerate(dataloader):
        ddp_input, ddp_target = batch.values()
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        step_model(model, input, target, accelerator, False)
        # Do "gradient accumulation" (noop)
        with accelerator.accumulate(ddp_model):
            step_model(ddp_model, ddp_input, ddp_target, accelerator)

        # DDP model and model should only be in sync when not (iteration % 2 == 0)
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not param.requires_grad:
                continue
            if ((iteration + 1) % 2 == 0) or (iteration == len(dataloader) - 1) or sync_each_batch:
                # Grads should be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is True
                ), f"Gradients not in sync when they should be at iteration {iteration}:\nModel grad ({param.grad}) != DDP grad ({ddp_param.grad})"
            else:
                # Grads should not be in sync
                assert (
                    torch.allclose(param.grad, ddp_param.grad) is False
                ), f"Gradients in sync when they should not be at iteration {iteration}:\nModel grad ({param.grad}) == DDP grad ({ddp_param.grad})"

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
        ddp_input = ddp_input[torch.randperm(len(ddp_input))]
    GradientState._reset_state()


def test_gradient_accumulation_with_opt_and_scheduler(
    split_batches=False, dispatch_batches=False, sync_each_batch=False
):
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2, sync_each_batch=sync_each_batch)
    dataloader_config = DataLoaderConfiguration(split_batches=split_batches, dispatch_batches=dispatch_batches)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    # Test that context manager behaves properly
    model, opt, sched, dataloader, ddp_model, ddp_opt, ddp_sched = get_training_setup(accelerator, True)
    for iteration, batch in enumerate(dataloader):
        ddp_input, ddp_target = batch.values()
        # Gather the distributed inputs and targs for the base model
        input, target = accelerator.gather((ddp_input, ddp_target))
        input, target = input.to(accelerator.device), target.to(accelerator.device)
        # Perform our initial ground truth step in non "DDP"
        model.train()
        ddp_model.train()
        step_model(model, input, target, accelerator, False)
        opt.step()

        if ((iteration + 1) % 2 == 0) or ((iteration + 1) == len(dataloader)):
            if split_batches:
                sched.step()
            else:
                for _ in range(accelerator.num_processes):
                    sched.step()

        # Perform gradient accumulation under wrapper
        with accelerator.accumulate(ddp_model):
            step_model(ddp_model, ddp_input, ddp_target, accelerator)
            ddp_opt.step()
            ddp_sched.step()

        # Learning rates should be the same
        assert (
            opt.param_groups[0]["lr"] == ddp_opt.param_groups[0]["lr"]
        ), f'Learning rates found in each optimizer did not align\nopt: {opt.param_groups[0]["lr"]}\nDDP opt: {ddp_opt.param_groups[0]["lr"]}\n'
        did_step = (((iteration + 1) % 2) == 0) or ((iteration + 1) == len(dataloader))
        if accelerator.num_processes > 1:
            check_model_parameters(
                model,
                ddp_model,
                did_step or sync_each_batch,  # syncs at each grad_accum interval of if sync_each_batch==True
                iteration,
                rtol=1e-3,  # needs a relative tolerance due to roundoff errors
            )

        if did_step:
            opt.zero_grad()  # flush gradients every accum step
        ddp_opt.zero_grad()

        # Shuffle ddp_input on each iteration
        torch.manual_seed(1337 + iteration)
    GradientState._reset_state()


def test_dataloader_break():
    accelerator = Accelerator()
    first_dset = RegressionDataset(length=80)
    first_dataloader = DataLoader(first_dset, batch_size=16)
    second_dset = RegressionDataset(length=96)
    second_dataloader = DataLoader(second_dset, batch_size=16)
    first_dataloader, second_dataloader = accelerator.prepare(first_dataloader, second_dataloader)

    assert accelerator.gradient_state.active_dataloader is None
    for iteration, _ in enumerate(first_dataloader):
        assert id(accelerator.gradient_state.active_dataloader) == id(first_dataloader)
        if iteration < len(first_dataloader) - 1:
            assert not accelerator.gradient_state.end_of_dataloader
            if iteration == 1:
                for batch_num, _ in enumerate(second_dataloader):
                    assert id(accelerator.gradient_state.active_dataloader) == id(second_dataloader)
                    if batch_num < len(second_dataloader) - 1:
                        assert not accelerator.gradient_state.end_of_dataloader
                    else:
                        assert accelerator.gradient_state.end_of_dataloader
        else:
            assert accelerator.gradient_state.end_of_dataloader
    assert accelerator.gradient_state.active_dataloader is None


def main():
    accelerator = Accelerator()
    state = accelerator.state
    if state.local_process_index == 0:
        print("**Test `accumulate` gradient accumulation with dataloader break**")
    if state.distributed_type != DistributedType.XLA:
        test_dataloader_break()
    if state.distributed_type == DistributedType.NO:
        if state.local_process_index == 0:
            print("**Test NOOP `no_sync` context manager**")
        test_noop_sync(accelerator)
    if state.distributed_type in (
        DistributedType.MULTI_GPU,
        DistributedType.MULTI_NPU,
        DistributedType.MULTI_MLU,
        DistributedType.MULTI_MUSA,
        DistributedType.MULTI_CPU,
    ):
        if state.local_process_index == 0:
            print("**Test Distributed `no_sync` context manager**")
        test_distributed_sync(accelerator)
        if state.local_process_index == 0:
            print("**Test Distributed `no_sync` context manager with multiple forwards**")
        test_distributed_sync_multiple_fwd(accelerator)
    if state.distributed_type in (
        DistributedType.MULTI_GPU,
        DistributedType.MULTI_NPU,
        DistributedType.MULTI_MLU,
        DistributedType.MULTI_MUSA,
    ):
        for split_batch in [True, False]:
            for dispatch_batches in [True, False]:
                for sync_each_batch in [True, False]:
                    if state.local_process_index == 0:
                        print(
                            "**Test `accumulate` gradient accumulation, ",
                            f"`split_batches={split_batch}` and `dispatch_batches={dispatch_batches}` and `sync_each_batch={sync_each_batch}`**",
                        )
                    test_gradient_accumulation(split_batch, dispatch_batches, sync_each_batch)

    # Currently will break on torch 2.0 +, need to investigate why
    if state.local_process_index == 0:
        print(
            "**Test `accumulate` gradient accumulation with optimizer and scheduler, ",
            "`split_batches=False`, `dispatch_batches=False`, `sync_each_batch=False`**",
        )
    test_gradient_accumulation_with_opt_and_scheduler()
    if state.distributed_type in (
        DistributedType.MULTI_GPU,
        DistributedType.MULTI_NPU,
        DistributedType.MULTI_MLU,
        DistributedType.MULTI_MUSA,
    ):
        for split_batch in [True, False]:
            for dispatch_batches in [True, False]:
                for sync_each_batch in [True, False]:
                    if not split_batch and not dispatch_batches and not sync_each_batch:
                        continue
                    if state.local_process_index == 0:
                        print(
                            "**Test `accumulate` gradient accumulation with optimizer and scheduler, ",
                            f"`split_batches={split_batch}` and `dispatch_batches={dispatch_batches}` and `sync_each_batch={sync_each_batch}`**",
                        )
                    test_gradient_accumulation_with_opt_and_scheduler(split_batch, dispatch_batches, sync_each_batch)
    state.destroy_process_group()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
