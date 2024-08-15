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

This particular script verifies this for DeepSpeed training.
"""
from unittest.mock import patch

from msamp import deepspeed
import evaluate
import torch
# import transformer_engine.common.recipe as te_recipe
# import transformer_engine.pytorch as te
from fp8_utils import evaluate_model, get_training_utilities
# from transformer_engine.common.recipe import DelayedScaling

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")


def train_baseline(zero_stage: int = 1, opt_level: str = "O1"):
    # This forces transformers to think Zero-3 Init should be used
    with patch("transformers.integrations.deepspeed.is_deepspeed_zero3_enabled") as mock:
        mock.return_value = zero_stage == 3
    set_seed(42)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    import numpy as np

    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {"device": "none", "nvme_path": None},
            "offload_param": {"device": "none", "nvme_path": None},
            "stage3_gather_16bit_weights_on_model_save": False,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": np.inf,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
        "msamp": {
            "enabled": True,
            "opt_level": opt_level,
        }
    }

    (
        model,
        optimizer,
        _,
        _,
    ) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=config,
    )

    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()

    for _ in range(2):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            for _ in range(accelerator.num_processes):
                lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.destroy()
    torch.cuda.empty_cache()
    AcceleratorState()._reset_state(True)
    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    return base_model_results, trained_model_results


def train_integration(zero_stage: int = 1):
    set_seed(42)
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    AcceleratorState()._reset_state(True)
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=zero_stage,
        zero3_init_flag=zero_stage == 3,
    )
    accelerator = Accelerator(
        mixed_precision="fp8", kwargs_handlers=kwargs_handlers, deepspeed_plugin=deepspeed_plugin
    )
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 16

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()
    model_outputs = []
    data = []
    for _ in range(2):
        for batch in train_dataloader:
            outputs = model(**batch)
            data.append(batch.to("cpu"))
            model_outputs.append(outputs.logits.to("cpu"))
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.destroy()
    torch.cuda.empty_cache()
    assert (
        trained_model_results["accuracy"] > base_model_results["accuracy"]
    ), f'Accuracy should be higher for the trained model: {trained_model_results["accuracy"]} > {base_model_results["accuracy"]}'
    assert (
        trained_model_results["f1"] > base_model_results["f1"]
    ), f'F1 score should be higher for the trained model: {trained_model_results["f1"]} > {base_model_results["f1"]}'

    return base_model_results, trained_model_results


if __name__ == "__main__":
    results = {"1": [], "2": [], "3": []}
    for zero_stage in [1, 2, 3]:
        for opt_level in ["O1", "O2", "O3"]:
            baseline_not_trained, baseline_trained = train_baseline(zero_stage, opt_level)
            results[zero_stage].append({"opt_level": opt_level, "not_trained": baseline_not_trained, "trained": baseline_trained})
    for stage, stage_results in results.items():
        print(f'zero_stage={stage}:\n')
        for result in stage_results:
            print(f'opt_level={result["opt_level"]}:\nBaseline not trained: {result["not_trained"]}\nBaseline trained: {result["trained"]}\n')
    # accelerator_not_trained, accelerator_trained, accelerator_outputs, accelerator_data = train_integration(zero_stage)
    # assert (
    #     baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"]
    # ), f'ZERO stage {zero_stage}: Accuracy should be the same for the baseline and accelerator: {baseline_not_trained["accuracy"]} == {accelerator_not_trained["accuracy"]}'
    # assert (
    #     baseline_not_trained["f1"] == accelerator_not_trained["f1"]
    # ), f'ZERO stage {zero_stage}: F1 score should be the same for the baseline and accelerator: {baseline_not_trained["f1"]} == {accelerator_not_trained["f1"]}'
    # assert (
    #     baseline_trained["accuracy"] == accelerator_trained["accuracy"]
    # ), f'ZERO stage {zero_stage}: Accuracy should be the same for the baseline and accelerator: {baseline_trained["accuracy"]} == {accelerator_trained["accuracy"]}'
    # assert (
    #     baseline_trained["f1"] == accelerator_trained["f1"]
    # ), f'ZERO stage {zero_stage}: F1 score should be the same for the baseline and accelerator: {baseline_trained["f1"]} == {accelerator_trained["f1"]}'

    torch.distributed.destroy_process_group()
