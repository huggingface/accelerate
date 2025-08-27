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
This script tests to ensure that `accelerate` performs at the same level as raw `torchao`.

This particular script verifies this for deepspeed training.
"""

from functools import partial
from unittest.mock import patch

import deepspeed
import evaluate
import torch
from fp8_utils import evaluate_model, get_training_utilities
from torchao.float8 import convert_to_float8_training
from transformers.integrations import HfDeepSpeedConfig

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.state import AcceleratorState
from accelerate.utils import AORecipeKwargs, set_seed


MODEL_NAME = "bert-base-cased"
METRIC = evaluate.load("glue", "mrpc")


def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    if fqn in (first_layer_name, last_layer_name):
        return False
    return True


def train_baseline(zero_stage: int = 1):
    set_seed(42)
    # This forces transformers to think Zero-3 Init should be used
    with patch("transformers.integrations.deepspeed.is_deepspeed_zero3_enabled") as mock:
        mock.return_value = zero_stage == 3

    config = HfDeepSpeedConfig(
        {
            "train_micro_batch_size_per_gpu": 16,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": zero_stage},
        }
    )
    plugin = DeepSpeedPlugin(hf_ds_config=config)
    accelerator = Accelerator(deepspeed_plugin=plugin)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )
    first_linear = None
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if first_linear is None:
                first_linear = name
            last_linear = name
    func = partial(filter_linear_layers, first_layer_name=first_linear, last_layer_name=last_linear)

    convert_to_float8_training(model, module_filter_fn=func)

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
    }

    (
        model,
        optimizer,
        _,
        lr_scheduler,
    ) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=config,
    )

    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()

    model_outputs = []
    data = []

    for batch in train_dataloader:
        outputs = model(**batch)
        data.append(batch.to("cpu"))
        model_outputs.append(outputs.logits.to("cpu"))
        loss = outputs.loss
        model.backward(loss)
        model.step()
        for _ in range(accelerator.num_processes):
            lr_scheduler.step()

    trained_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.destroy()
    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    del config
    return base_model_results, trained_model_results, model_outputs, data


def train_integration(zero_stage: int = 1):
    set_seed(42)
    AcceleratorState()._reset_state(True)
    config = HfDeepSpeedConfig(
        {
            "train_micro_batch_size_per_gpu": 16,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": zero_stage},
        }
    )
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=config,
    )
    # This forces transformers to think Zero-3 Init should be used
    with patch("transformers.integrations.deepspeed.is_deepspeed_zero3_enabled") as mock:
        mock.return_value = zero_stage == 3
    accelerator = Accelerator(
        mixed_precision="fp8", kwargs_handlers=[AORecipeKwargs()], deepspeed_plugin=deepspeed_plugin
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )

    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    )
    base_model_results = evaluate_model(model, eval_dataloader, METRIC, accelerator=accelerator)
    model.train()
    model_outputs = []
    data = []
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
    assert trained_model_results["accuracy"] > base_model_results["accuracy"], (
        f"Accuracy should be higher for the trained model: {trained_model_results['accuracy']} > {base_model_results['accuracy']}"
    )
    assert trained_model_results["f1"] > base_model_results["f1"], (
        f"F1 score should be higher for the trained model: {trained_model_results['f1']} > {base_model_results['f1']}"
    )

    del config
    return base_model_results, trained_model_results, model_outputs, data


if __name__ == "__main__":
    for zero_stage in [1, 2, 3]:
        baseline_not_trained, baseline_trained, baseline_outputs, baseline_data = train_baseline(zero_stage)
        accelerator_not_trained, accelerator_trained, accelerator_outputs, accelerator_data = train_integration(
            zero_stage
        )
        assert baseline_not_trained["accuracy"] == accelerator_not_trained["accuracy"], (
            f"ZERO stage {zero_stage}: Accuracy should be the same for the baseline and accelerator: {baseline_not_trained['accuracy']} == {accelerator_not_trained['accuracy']}"
        )
        assert baseline_not_trained["f1"] == accelerator_not_trained["f1"], (
            f"ZERO stage {zero_stage}: F1 score should be the same for the baseline and accelerator: {baseline_not_trained['f1']} == {accelerator_not_trained['f1']}"
        )
        assert baseline_trained["accuracy"] == accelerator_trained["accuracy"], (
            f"ZERO stage {zero_stage}: Accuracy should be the same for the baseline and accelerator: {baseline_trained['accuracy']} == {accelerator_trained['accuracy']}"
        )
        assert baseline_trained["f1"] == accelerator_trained["f1"], (
            f"ZERO stage {zero_stage}: F1 score should be the same for the baseline and accelerator: {baseline_trained['f1']} == {accelerator_trained['f1']}"
        )
        AcceleratorState()._reset_state(True)
    torch.distributed.destroy_process_group()
