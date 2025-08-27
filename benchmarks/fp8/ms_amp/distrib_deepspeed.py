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

NOTE: MS-AMP does *not* support ZeRO-3.
"""

# import msamp.deepspeed as msamp_deepspeed
import evaluate
import torch
from fp8_utils import evaluate_model, get_training_utilities
from msamp import deepspeed as msamp_deepspeed

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")


def train_baseline(zero_stage: int = 1, opt_level: str = "O1"):
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
        },
        "gradient_clipping": 1.0,
        "steps_per_print": np.inf,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
        "msamp": {
            "enabled": True,
            "opt_level": opt_level,
        },
    }
    (
        model,
        optimizer,
        _,
        _,
    ) = msamp_deepspeed.initialize(
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
    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    return base_model_results, trained_model_results


def train_integration(zero_stage: int = 1, opt_level: str = "O1"):
    set_seed(42)
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=zero_stage,
        enable_msamp=True,
        msamp_opt_level=opt_level,
    )
    accelerator = Accelerator(mixed_precision="fp8", deepspeed_plugin=deepspeed_plugin)
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 16

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()
    for _ in range(2):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.destroy()
    torch.cuda.empty_cache()
    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    AcceleratorState()._reset_state(True)
    return base_model_results, trained_model_results


if __name__ == "__main__":
    for zero_stage in [1, 2]:
        for opt_level in ["O1", "O2", "O3"]:
            baseline_not_trained, baseline_trained = train_baseline(zero_stage, opt_level)
            accelerator_not_trained, accelerator_trained = train_integration(zero_stage, opt_level)
            assert baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"], (
                f"ZERO stage {zero_stage}, opt_level={opt_level}:\nAccuracy should be the same for the baseline and accelerator: {baseline_not_trained['accuracy']} == {accelerator_not_trained['accuracy']}"
            )
            assert baseline_not_trained["f1"] == accelerator_not_trained["f1"], (
                f"ZERO stage {zero_stage}, opt_level={opt_level}:\nF1 score should be the same for the baseline and accelerator: {baseline_not_trained['f1']} == {accelerator_not_trained['f1']}"
            )
            assert baseline_trained["accuracy"] == accelerator_trained["accuracy"], (
                f"ZERO stage {zero_stage}, opt_level={opt_level}:\nAccuracy should be the same for the baseline and accelerator: {baseline_trained['accuracy']} == {accelerator_trained['accuracy']}"
            )
            assert baseline_trained["f1"] == accelerator_trained["f1"], (
                f"ZERO stage {zero_stage}, opt_level={opt_level}:\nF1 score should be the same for the baseline and accelerator: {baseline_trained['f1']} == {accelerator_trained['f1']}"
            )

    torch.distributed.destroy_process_group()
