# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest

import torch
from parameterized import parameterized

from accelerate.big_modeling import _attach_sequence_parallel_hooks
from accelerate.test_utils.testing import (
    TempDirTestCase,
    execute_subprocess_async,
    get_launch_command,
    path_in_accelerate_package,
    require_fsdp2,
    require_multi_device,
    require_non_torch_xla,
    require_transformers,
    run_first,
)
from accelerate.utils import patch_environment
from accelerate.utils.imports import _is_package_available, is_transformers_available
from accelerate.utils.ulysses import _packed_causal_mask


if is_transformers_available():
    from transformers import AutoConfig, AutoModelForCausalLM


class _FakeMesh:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size

    def get_local_rank(self):
        return 0

    def get_group(self):
        return None


class TorchUlyssesSPUnitTest(unittest.TestCase):
    def test_packed_causal_mask(self):
        # Two documents of 3 and 2 tokens: causal within a document, nothing across.
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        mask = _packed_causal_mask(position_ids)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        assert mask.shape == (1, 1, 5, 5)
        assert torch.equal(mask[0, 0], expected)

    def test_packed_causal_mask_is_none_without_packing(self):
        assert _packed_causal_mask(torch.arange(8).unsqueeze(0)) is None
        assert _packed_causal_mask(torch.arange(4, 12).unsqueeze(0)) is None


@require_transformers
class TorchUlyssesSPModelChecks(unittest.TestCase):
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def _model(self, attn_implementation="sdpa", **config_overrides):
        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation)
        # Llama has no `layer_types`, so set the overrides once the config is built
        for name, value in config_overrides.items():
            setattr(model.config, name, value)
        return model

    def test_refuses_eager(self):
        with self.assertRaisesRegex(ValueError, "does not support the `eager` attention implementation"):
            _attach_sequence_parallel_hooks(self._model("eager"), _FakeMesh(2))

    def test_refuses_indivisible_heads(self):
        # tiny-random-Llama has 4 attention heads and 4 kv heads
        with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
            _attach_sequence_parallel_hooks(self._model(), _FakeMesh(8))

    def test_refuses_sliding_window_with_sdpa(self):
        model = self._model(layer_types=["sliding_attention", "full_attention"], sliding_window=8)
        with self.assertRaisesRegex(ValueError, "with `sdpa` does not support attention layers of type"):
            _attach_sequence_parallel_hooks(model, _FakeMesh(2))

    def test_refuses_chunked_attention(self):
        model = self._model(layer_types=["chunked_attention", "full_attention"])
        with self.assertRaisesRegex(ValueError, "layers of type \\['chunked_attention'\\]"):
            _attach_sequence_parallel_hooks(model, _FakeMesh(2))

    def test_refuses_non_attention_sequence_mixing_layers(self):
        # LFM2-style short convolutions run on the local shard and never see the gathered sequence
        model = self._model("kernels-community/flash-attn2", layer_types=["conv", "full_attention"])
        with self.assertRaisesRegex(ValueError, "layers of type \\['conv'\\]"):
            _attach_sequence_parallel_hooks(model, _FakeMesh(2))


@require_fsdp2
@run_first
@require_non_torch_xla
@require_multi_device
@require_transformers
class TorchUlyssesSPIntegrationTest(TempDirTestCase):
    test_scripts_folder = path_in_accelerate_package("test_utils", "scripts", "external_deps")

    def _launch(self, num_processes, *script_args):
        cmd = get_launch_command(num_processes=num_processes, num_machines=1, machine_rank=0)
        cmd.extend([str(self.test_scripts_folder / "test_torch_ulysses_sp.py"), *script_args])
        with patch_environment(omp_num_threads=1):
            execute_subprocess_async(cmd)

    @parameterized.expand([(False,), (True,)])
    def test_sdpa_matches_reference(self, packed):
        args = ["--sp_size=2", "--attn_implementation=sdpa"]
        if packed:
            args.append("--packed")
        self._launch(2, *args)

    @unittest.skipUnless(
        _is_package_available("kernels"), "test requires the kernels library for the hub flash-attn2 kernel"
    )
    @parameterized.expand([(False,), (True,)])
    def test_flash_attention_matches_reference(self, packed):
        args = ["--sp_size=2", "--attn_implementation=kernels-community/flash-attn2", "--dtype=bfloat16"]
        if packed:
            args.append("--packed")
        self._launch(2, *args)

    @unittest.skipUnless(
        _is_package_available("kernels"), "test requires the kernels library for the hub flash-attn2 kernel"
    )
    def test_flash_attention_ignores_stale_sequence_boundaries(self):
        """Boundaries a padding-free collator computed for the local shard must not reach the gathered attention."""
        self._launch(
            2,
            "--sp_size=2",
            "--attn_implementation=kernels-community/flash-attn2",
            "--dtype=bfloat16",
            "--packed",
            "--stale_flash_attn_kwargs",
        )

    @unittest.skipUnless(
        _is_package_available("kernels"), "test requires the kernels library for the hub flash-attn2 kernel"
    )
    def test_flash_attention_applies_sliding_window(self):
        """Flash attention applies the window over the gathered sequence: the local attention call is intact."""
        self._launch(
            2,
            "--sp_size=2",
            "--attn_implementation=kernels-community/flash-attn2",
            "--dtype=bfloat16",
            "--model_name_or_path=hf-internal-testing/tiny-random-MistralForCausalLM",
            "--sliding_window=16",
        )

    @unittest.skipUnless(
        _is_package_available("kernels"), "test requires the kernels library for the hub flash-attn2 kernel"
    )
    def test_attention_sinks(self):
        """gpt-oss: one sink logit per head, sliced to the heads each rank holds after the all-to-all."""
        self._launch(
            2,
            "--sp_size=2",
            "--attn_implementation=kernels-community/flash-attn2",
            "--dtype=bfloat16",
            "--model_name_or_path=trl-internal-testing/tiny-GptOssForCausalLM",
        )

    def test_vision_language_model_on_text(self):
        """Qwen2.5-VL: text-only training, attention layers get no `position_ids` and use the model-level ones."""
        self._launch(
            2,
            "--sp_size=2",
            "--attn_implementation=sdpa",
            "--model_name_or_path=trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        )

    def test_sp_composes_with_fsdp(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("test requires 4 devices for dp_shard_size=2 x sp_size=2")
        self._launch(4, "--sp_size=2", "--dp_shard_size=2")
