"""Reproduction script for FSDP2 auto-wrap policy bug.

When using FSDP2 with `TRANSFORMER_BASED_WRAP` and without explicitly setting
`fsdp_transformer_layer_cls_to_wrap`, the auto-wrap policy returns False for all
modules, causing the entire model to be treated as a single FSDP unit.

The root cause is in `fsdp2_prepare_auto_wrap_policy`: the `_no_split_modules`
fallback correctly populates a local `transformer_cls_to_wrap` set, but the
inner `policy()` closure checks `fsdp2_plugin.transformer_cls_names_to_wrap`
(the plugin attribute, which is None) instead of the local variable.

This means per-layer wrapping never occurs when relying on `_no_split_modules`,
which is the default for most HuggingFace models.

Impact: without per-layer wrapping, every GPU all-gathers the entire model's
parameters during forward, causing OOM on large models that would otherwise fit.

Run: python reproduce_fsdp2_wrap_bug.py
(No GPU required.)
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from accelerate.utils.fsdp_utils import fsdp2_prepare_auto_wrap_policy


def test_fsdp2_auto_wrap_policy():
    # Use a small public model (no download needed with meta device)
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    decoder_cls_name = model._no_split_modules[0]
    layer_count = sum(1 for m in model.modules() if m.__class__.__name__ == decoder_cls_name)
    print(f"Model: SmolLM2-135M, _no_split_modules: {model._no_split_modules}, layers: {layer_count}")

    # Create plugin with TRANSFORMER_BASED_WRAP but NO explicit cls_names_to_wrap
    # (the common case - relying on _no_split_modules from the model)
    plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        fsdp_version=2,
    )
    assert plugin.transformer_cls_names_to_wrap is None, "Expected None (relying on _no_split_modules fallback)"

    policy_func = fsdp2_prepare_auto_wrap_policy(plugin, model)
    assert policy_func is not None, "policy_func should not be None"

    matched = sum(1 for m in model.modules() if policy_func(m))
    print(f"Modules matching policy: {matched} (expected: {layer_count})")

    assert matched == layer_count, (
        f"BUG: policy matched {matched} modules, expected {layer_count}. "
        "The _no_split_modules fallback is being ignored by the policy closure."
    )
    print("PASSED: All decoder layers would be wrapped correctly.")


if __name__ == "__main__":
    test_fsdp2_auto_wrap_policy()
