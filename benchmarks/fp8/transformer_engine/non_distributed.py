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
This script tests to ensure that `accelerate` performs at the same level as raw `TransformersEngine`.

This particular script verifies this for single GPU training.
"""

import evaluate
import torch
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from fp8_utils import evaluate_model, get_named_parameters, get_training_utilities
from transformer_engine.common.recipe import DelayedScaling

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed
from accelerate.utils.transformer_engine import convert_model


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")


def train_baseline():
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(MODEL_NAME)

    # Convert the model to TE
    old_named_params = get_named_parameters(model)

    with torch.no_grad():
        convert_model(model)

    new_named_params = get_named_parameters(model)
    mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
    for param_group in optimizer.param_groups:
        param_group["params"] = [mapping[p] for p in param_group["params"]]

    FP8_RECIPE_KWARGS = {"fp8_format": te_recipe.Format.HYBRID, "amax_history_len": 32, "amax_compute_algo": "max"}
    fp8_recipe = DelayedScaling(**FP8_RECIPE_KWARGS)

    model.to("cuda")
    base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()

    for batch in train_dataloader:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = batch.to("cuda")
                outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC)

    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    return base_model_results, trained_model_results


def train_integration():
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    AcceleratorState()._reset_state(True)
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC)
    model.train()

    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC)

    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    return base_model_results, trained_model_results


if __name__ == "__main__":
    baseline_not_trained, baseline_trained = train_baseline()
    accelerator_not_trained, accelerator_trained = train_integration()

    assert baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"], (
        f"Accuracy should be the same for the baseline and accelerator: {baseline_not_trained['accuracy']} == {accelerator_not_trained['accuracy']}"
    )
    assert baseline_not_trained["f1"] == accelerator_not_trained["f1"], (
        f"F1 score should be the same for the baseline and accelerator: {baseline_not_trained['f1']} == {accelerator_not_trained['f1']}"
    )
    assert baseline_trained["accuracy"] == accelerator_trained["accuracy"], (
        f"Accuracy should be the same for the baseline and accelerator: {baseline_trained['accuracy']} == {accelerator_trained['accuracy']}"
    )
    assert baseline_trained["f1"] == accelerator_trained["f1"], (
        f"F1 score should be the same for the baseline and accelerator: {baseline_trained['f1']} == {accelerator_trained['f1']}"
    )
