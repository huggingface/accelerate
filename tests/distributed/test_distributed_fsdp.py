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

import json
import os
import tempfile

from parameterized import parameterized

from accelerate.test_utils.testing import require_huggingface_suite, require_multi_gpu, slow

from .test_distributed import CONFIGS_DIR, DistributedTrainingCommon, train


FSDP2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp2.yaml")
FSDP2_ACTIVATION_CHECKPOINTING_COMPILE_CONFIG_FILE = os.path.join(
    CONFIGS_DIR, "fsdp2_activation_checkpointing_compile.yaml"
)


@slow
@require_multi_gpu
@require_huggingface_suite
class FSDP2TrainingTest(DistributedTrainingCommon):
    @parameterized.expand(["fp32", "bf16", "fp16"])
    def test_training(self, dtype):
        self.check_training(dtype, config_file=FSDP2_CONFIG_FILE)

    @parameterized.expand(["fp16", "bf16"])
    def test_training_mixed_precision(self, mixed_precision):
        self.check_mixed_precision(mixed_precision, config_file=FSDP2_CONFIG_FILE)

    def test_training_with_gradient_accumulation(self):
        self.check_gradient_accumulation(config_file=FSDP2_CONFIG_FILE)

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=FSDP2_CONFIG_FILE)

    def test_training_with_activation_checkpointing_and_regional_compilation(self):
        """Checkpointed FSDP2 training with regional compilation should match the unsharded reference."""
        reference_losses = train(dtype="fp32")
        with tempfile.TemporaryDirectory() as output_dir:
            compiled_losses = train(
                FSDP2_ACTIVATION_CHECKPOINTING_COMPILE_CONFIG_FILE, output_dir=output_dir, dtype="fp32"
            )
            with open(os.path.join(output_dir, "compile_stats.json")) as f:
                compile_stats = json.load(f)

        self.assert_same_training(compiled_losses, reference_losses)

        # These observations come from rank 0. Require wrappers so the counts cannot all be zero.
        self.assertGreater(compile_stats["checkpoint_wrappers"], 0)
        self.assertEqual(compile_stats["checkpoint_wrappers_sharded"], compile_stats["checkpoint_wrappers"])
        self.assertEqual(compile_stats["inner_modules_sharded"], 0)

        # Other regions can compile even if regional compilation skips the checkpointed blocks.
        self.assertGreater(compile_stats["compiled_graphs"], 0)
        self.assertEqual(
            compile_stats["checkpoint_wrappers_with_compiled_callable"], compile_stats["checkpoint_wrappers"]
        )

        # A correct FSDP layout alone does not rule out breaks involving a checkpoint wrapper.
        self.assertEqual(compile_stats["graph_breaks_with_checkpoint_wrapper"], 0, compile_stats)

    def test_training_with_activation_checkpointing_and_regional_compilation_can_resume(self):
        self.check_resume(config_file=FSDP2_ACTIVATION_CHECKPOINTING_COMPILE_CONFIG_FILE)
