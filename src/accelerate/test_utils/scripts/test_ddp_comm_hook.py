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
import torch

from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs, PartialState
from accelerate.utils import is_hpu_available


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.p = torch.nn.Parameter(torch.randn(40, 20))

    def forward(self, x, rank):
        return self.p * (x ** (1 + rank))


def _run_and_get_grads(model, rank):
    torch.manual_seed(2024)
    input = torch.randn(40, 20)
    output = model(input, rank)
    output.mean().backward()
    param = next(model.parameters())
    return param.grad


def test_ddp_comm_hook(comm_hook, comm_wrapper, comm_state_option):
    ddp_kwargs = DistributedDataParallelKwargs(
        comm_hook=comm_hook,
        comm_wrapper=comm_wrapper,
        comm_state_option=comm_state_option,
    )
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    model = accelerator.prepare(MockModel())
    hook_grads = _run_and_get_grads(model, accelerator.local_process_index)

    reference_model = torch.nn.parallel.DistributedDataParallel(
        MockModel().to(accelerator.device),
        device_ids=[accelerator.local_process_index],
        output_device=accelerator.local_process_index,
    )
    reference_grads = _run_and_get_grads(reference_model, accelerator.local_process_index)

    torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-2, atol=1e-2)


def main():
    for comm_hook, comm_wrapper, comm_state_option in [
        (DDPCommunicationHookType.NO, DDPCommunicationHookType.NO, {}),
        (DDPCommunicationHookType.FP16, DDPCommunicationHookType.NO, {}),
        (DDPCommunicationHookType.BF16, DDPCommunicationHookType.NO, {}),
        (DDPCommunicationHookType.POWER_SGD, DDPCommunicationHookType.NO, {}),
        (DDPCommunicationHookType.POWER_SGD, DDPCommunicationHookType.FP16, {}),
        (DDPCommunicationHookType.POWER_SGD, DDPCommunicationHookType.BF16, {}),
        (DDPCommunicationHookType.POWER_SGD, DDPCommunicationHookType.NO, {"matrix_approximation_rank": 2}),
        (DDPCommunicationHookType.BATCHED_POWER_SGD, DDPCommunicationHookType.NO, {}),
        (DDPCommunicationHookType.BATCHED_POWER_SGD, DDPCommunicationHookType.FP16, {}),
        (DDPCommunicationHookType.BATCHED_POWER_SGD, DDPCommunicationHookType.BF16, {}),
    ]:
        if is_hpu_available():
            HPU_UNSUPPORTED_COMM_HOOKS = {DDPCommunicationHookType.FP16, DDPCommunicationHookType.BF16}
            if comm_hook in HPU_UNSUPPORTED_COMM_HOOKS or comm_wrapper in HPU_UNSUPPORTED_COMM_HOOKS:
                print(f"Skipping test DDP comm hook: {comm_hook}, comm wrapper: {comm_wrapper} on HPU")
                continue

        print(f"Test DDP comm hook: {comm_hook}, comm wrapper: {comm_wrapper}")
        test_ddp_comm_hook(comm_hook, comm_wrapper, comm_state_option)
    PartialState().destroy_process_group()


if __name__ == "__main__":
    main()
