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
import torch
from msamp.fsdp import FsdpReplacer, FP8FullyShardedDataParallel
from msamp.optim import FSDPAdamW
from fp8_utils import evaluate_model, get_training_utilities, get_named_parameters, get_dataloaders

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from transformers.models.bert import BertLayer
from functools import partial


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")
FSDP_WRAP_POLICY = partial(transformer_auto_wrap_policy, transformer_layer_cls={BertLayer})


from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

def train_baseline(opt_level="O2"):
    set_seed(42)
    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    train_dataloader, eval_dataloader = get_dataloaders(MODEL_NAME)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # old_named_params = get_named_parameters(model)
    # This single call:
    # 1. Replaces all linear layers with MS-AMP's `LinearReplacer`
    # 2. Replaces the weights with `ScalingParameters`
    model.to(device)
    model = FsdpReplacer.replace(model)

    # Same as FullyShardedDataParallel, but overrides `FlatParamHandle`, `post_backward_hook`, and adds comm hook
    model = FP8FullyShardedDataParallel(
        model,
        use_orig_params=True,
        auto_wrap_policy=FSDP_WRAP_POLICY,
    )

    # TODO: Make this happen using existing AdamW
    optimizer = FSDPAdamW(
        model.parameters(),
        lr=0.0001,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * 2,
    )

    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()

    for i, batch in enumerate(train_dataloader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)

    print(f'Process {accelerator.process_index}:\nBase model results: {base_model_results}\nTrained model results: {trained_model_results}')
    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

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
