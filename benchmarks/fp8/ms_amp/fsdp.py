# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
This script tests to ensure that `accelerate` performs at the same level as raw `MS-AMP`.

This particular script verifies this for FSDP training.
"""
import evaluate
import msamp
import inspect
import torch
from msamp.fsdp import FsdpReplacer, FP8FullyShardedDataParallel
from msamp.optim import FSDPAdamW, LBAdamW
from fp8_utils import evaluate_model, get_training_utilities, get_named_parameters, get_dataloaders

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from transformers.models.bert import BertLayer
from functools import partial
import torch.distributed as dist
from msamp.common.tensor import ScalingMeta, ScalingTensor


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")
FSDP_WRAP_POLICY = partial(transformer_auto_wrap_policy, transformer_layer_cls={BertLayer})


from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor


class MSAMPOptimWrapper(torch.optim.Optimizer):
    """
    Wrapper around an optimizer to make it compatible for FSDP.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.adjust_param_groups()

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        for param in self.original_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    def step(self):
        for i, param in enumerate(self.original_params):
            if self.master_weights[i] is not None:
                grad_meta = param._grad_meta
                dtype = Dtypes.qtype_to_dtype[grad_meta.qtype]
                self.master_weights[i].grad = ScalingTensor(param.grad.view(dtype), grad_meta)
                param.grad = None
        self.optimizer.step()

        # Copy master weight to weight
        for i, param in enumerate(self.original_params):
            if hasattr(param, '_meta') and param._meta is not None:
                hp_data = None
                if param.numel() == 0:
                    param._meta.amax[0].zero_()
                else:
                    hp_data = self.master_weights[i].float()
                    param._meta.amax[0] = hp_data.abs().max()

                dist.all_reduce(param._meta.amax[0], op=dist.ReduceOp.MAX)
                param._meta.reset_scaling_factor()
                if param.numel() > 0:
                    with ScalingMeta.in_time_scaling_context(False):
                        data = hp_data.cast(param._meta.qtype, param._meta, False) \
                                .value.view(torch.float32)
                    param.data.copy_(data)
                else:
                    param._meta.scale_inv.data.copy_(torch.reciprocal(param._meta.scale))

    def train(self):
        """
        Sets the optimizer to "train" mode. Useful for optimizers like `schedule_free`
        """
        return self.optimizer.train()

    def eval(self):
        """
        Sets the optimizer to "eval" mode. Useful for optimizers like `schedule_free`
        """
        return self.optimizer.eval()

    def adjust_param_groups(self):
        self.original_params, self.master_weights = [], []
        for group in self.param_groups:
            params = []
            for param in group['params']:
                if param is None:
                    continue

                self.original_params.append(param)
                if hasattr(param, '_meta') and param._meta is not None and param.numel() > 0:
                    dtype = Dtypes.qtype_to_dtype[param._meta.qtype]
                    param = ScalingTensor(param.view(dtype), param._meta)
                    master_weight = param.cast(Dtypes.kfloat16)
                    master_weight.requires_grad = True
                    self.master_weights.append(master_weight)
                    params.append(master_weight)
                else:
                    self.master_weights.append(None)
                    params.append(param)

            group['params'] = params



def train_baseline(opt_level="O2"):
    set_seed(42)
    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    train_dataloader, eval_dataloader = get_dataloaders(MODEL_NAME)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    from msamp.nn import LinearReplacer
    model = LinearReplacer.replace(model, weight_qtype=Dtypes.kfloat8_e4m3)

    for _, submodule in model.named_modules():
        params_to_process = list(submodule.named_parameters(recurse=False))
        for param_name, param in params_to_process:
            if not isinstance(param, torch.Tensor):
                data = param.value.view(-1)
                padded = 0
                if data.numel() % 4 != 0:
                    padded = 4 - data.numel() % 4
                    data = torch.nn.functional.pad(data, (0, padded))

                data = data.view(dtype=torch.float32)
                new_param = torch.nn.Parameter(data)
                new_param._original_shape = param.shape
                new_param._padded = padded
                new_param._meta = param.meta
                new_param._scaling_metas = param._scaling_metas

                setattr(submodule, param_name, new_param)

    model.to(device)
    model = FP8FullyShardedDataParallel(
        model,
        use_orig_params=True,
        auto_wrap_policy=FSDP_WRAP_POLICY,
    )

    # optimizer = FSDPAdamW(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    default_args = optimizer.defaults

    default_args['exp_avg_dtype'] = torch.uint8
    default_args['exp_avg_sq_dtype'] = torch.float16

    # Currently, we don't support foreach, capturable, differentiable, and fused.
    for k in ['foreach', 'capturable', 'differentiable', 'fused']:
        default_args.pop(k, None)

    optimizer = LBAdamW(optimizer.param_groups, **default_args)

    optimizer = MSAMPOptimWrapper(optimizer)
    # Same as FullyShardedDataParallel, but overrides `FlatParamHandle`, `post_backward_hook`, and adds comm hook

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * 2,
    )

    # base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()

    for i, batch in enumerate(train_dataloader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    # trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)

    # print(f'Process {accelerator.process_index}:\nBase model results: {base_model_results}\nTrained model results: {trained_model_results}')
    # assert (
    #     trained_model_results["accuracy"] > base_model_results["accuracy"]
    # ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    # assert (
    #     trained_model_results["f1"] > base_model_results["f1"]
    # ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    # return base_model_results, trained_model_results


def train_integration(opt_level="O2"):
    kwargs_handlers = [FP8RecipeKwargs(backend="msamp", opt_level=opt_level)]
    AcceleratorState()._reset_state(True)
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()
    for i, batch in enumerate(train_dataloader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)

    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    return base_model_results, trained_model_results


if __name__ == "__main__":
    # for opt_level in ["O1", "O2"]:
    train_baseline()
        # accelerator_not_trained, accelerator_trained = train_integration(opt_level)
        # assert (
        #     baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"]
        # ), f'Accuracy not the same for untrained baseline and accelerator using opt_level={opt_level}: {baseline_not_trained["accuracy"]} == {accelerator_not_trained["accuracy"]}'
        # assert (
        #     baseline_not_trained["f1"] == accelerator_not_trained["f1"]
        # ), f'F1 not the same for untrained baseline and accelerator using opt_level={opt_level}: {baseline_not_trained["f1"]} == {accelerator_not_trained["f1"]}'
        # assert (
        #     baseline_trained["accuracy"] == accelerator_trained["accuracy"]
        # ), f'Accuracy not the same for trained baseline and accelerator using opt_level={opt_level}: {baseline_trained["accuracy"]} == {accelerator_trained["accuracy"]}'
        # assert (
        #     baseline_trained["f1"] == accelerator_trained["f1"]
        # ), f'F1 not the same for trained baseline and accelerator using opt_level={opt_level}: {baseline_trained["f1"]} == {accelerator_trained["f1"]}'
