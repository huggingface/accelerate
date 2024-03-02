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

import ast
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from accelerate.test_utils.examples import compare_against_test
from accelerate.test_utils.testing import (
    TempDirTestCase,
    get_launch_command,
    require_huggingface_suite,
    require_multi_gpu,
    require_pippy,
    require_trackers,
    run_command,
    slow,
)
from accelerate.utils import write_basic_config


# DataLoaders built from `test_samples/MRPC` for quick testing
# Should mock `{script_name}.get_dataloaders` via:
# @mock.patch("{script_name}.get_dataloaders", mocked_dataloaders)

EXCLUDE_EXAMPLES = [
    "cross_validation.py",
    "gradient_accumulation.py",
    "local_sgd.py",
    "multi_process_metrics.py",
    "memory.py",
    "automatic_gradient_accumulation.py",
    "fsdp_with_peak_mem_tracking.py",
    "deepspeed_with_config_support.py",
    "megatron_lm_gpt_pretraining.py",
    "early_stopping.py",
]


class ExampleDifferenceTests(unittest.TestCase):
    """
    This TestCase checks that all of the `complete_*` scripts contain all of the
    information found in the `by_feature` scripts, line for line. If one fails,
    then a complete example does not contain all of the features in the features
    scripts, and should be updated.

    Each example script should be a single test (such as `test_nlp_example`),
    and should run `one_complete_example` twice: once with `parser_only=True`,
    and the other with `parser_only=False`. This is so that when the test
    failures are returned to the user, they understand if the discrepancy lies in
    the `main` function, or the `training_loop` function. Otherwise it will be
    unclear.

    Also, if there are any expected differences between the base script used and
    `complete_nlp_example.py` (the canonical base script), these should be included in
    `special_strings`. These would be differences in how something is logged, print statements,
    etc (such as calls to `Accelerate.log()`)
    """

    by_feature_path = Path("examples", "by_feature").resolve()
    examples_path = Path("examples").resolve()

    def one_complete_example(
        self, complete_file_name: str, parser_only: bool, secondary_filename: str = None, special_strings: list = None
    ):
        """
        Tests a single `complete` example against all of the implemented `by_feature` scripts

        Args:
            complete_file_name (`str`):
                The filename of a complete example
            parser_only (`bool`):
                Whether to look at the main training function, or the argument parser
            secondary_filename (`str`, *optional*):
                A potential secondary base file to strip all script information not relevant for checking,
                such as "cv_example.py" when testing "complete_cv_example.py"
            special_strings (`list`, *optional*):
                A list of strings to potentially remove before checking no differences are left. These should be
                diffs that are file specific, such as different logging variations between files.
        """
        self.maxDiff = None
        for item in os.listdir(self.by_feature_path):
            if item not in EXCLUDE_EXAMPLES:
                item_path = self.by_feature_path / item
                if item_path.is_file() and item_path.suffix == ".py":
                    with self.subTest(
                        tested_script=complete_file_name,
                        feature_script=item,
                        tested_section="main()" if parser_only else "training_function()",
                    ):
                        diff = compare_against_test(
                            self.examples_path / complete_file_name, item_path, parser_only, secondary_filename
                        )
                        diff = "\n".join(diff)
                        if special_strings is not None:
                            for string in special_strings:
                                diff = diff.replace(string, "")
                        assert diff == ""

    def test_nlp_examples(self):
        self.one_complete_example("complete_nlp_example.py", True)
        self.one_complete_example("complete_nlp_example.py", False)

    def test_cv_examples(self):
        cv_path = (self.examples_path / "cv_example.py").resolve()
        special_strings = [
            " " * 16 + "{\n\n",
            " " * 20 + '"accuracy": eval_metric["accuracy"],\n\n',
            " " * 20 + '"f1": eval_metric["f1"],\n\n',
            " " * 20 + '"train_loss": total_loss.item() / len(train_dataloader),\n\n',
            " " * 20 + '"epoch": epoch,\n\n',
            " " * 16 + "},\n\n",
            " " * 16 + "step=epoch,\n",
            " " * 12,
            " " * 8 + "for step, batch in enumerate(active_dataloader):\n",
        ]
        self.one_complete_example("complete_cv_example.py", True, cv_path, special_strings)
        self.one_complete_example("complete_cv_example.py", False, cv_path, special_strings)


@mock.patch.dict(os.environ, {"TESTING_MOCKED_DATALOADERS": "1"})
@require_huggingface_suite
class FeatureExamplesTests(TempDirTestCase):
    clear_on_setup = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.config_file = Path(cls._tmpdir) / "default_config.yml"

        write_basic_config(save_location=cls.config_file)
        cls.launch_args = get_launch_command(config_file=cls.config_file)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)

    def test_checkpointing_by_epoch(self):
        testargs = f"""
        examples/by_feature/checkpointing.py
        --checkpointing_steps epoch
        --output_dir {self.tmpdir}
        """.split()
        run_command(self.launch_args + testargs)
        assert (self.tmpdir / "epoch_0").exists()

    def test_checkpointing_by_steps(self):
        testargs = f"""
        examples/by_feature/checkpointing.py
        --checkpointing_steps 1
        --output_dir {self.tmpdir}
        """.split()
        _ = run_command(self.launch_args + testargs)
        assert (self.tmpdir / "step_2").exists()

    def test_load_states_by_epoch(self):
        testargs = f"""
        examples/by_feature/checkpointing.py
        --resume_from_checkpoint {self.tmpdir / "epoch_0"}
        """.split()
        output = run_command(self.launch_args + testargs, return_stdout=True)
        assert "epoch 0:" not in output
        assert "epoch 1:" in output

    def test_load_states_by_steps(self):
        testargs = f"""
        examples/by_feature/checkpointing.py
        --resume_from_checkpoint {self.tmpdir / "step_2"}
        """.split()
        output = run_command(self.launch_args + testargs, return_stdout=True)
        if torch.cuda.is_available():
            num_processes = torch.cuda.device_count()
        else:
            num_processes = 1
        if num_processes > 1:
            assert "epoch 0:" not in output
            assert "epoch 1:" in output
        else:
            assert "epoch 0:" in output
            assert "epoch 1:" in output

    @slow
    def test_cross_validation(self):
        testargs = """
        examples/by_feature/cross_validation.py
        --num_folds 2
        """.split()
        with mock.patch.dict(os.environ, {"TESTING_MOCKED_DATALOADERS": "0"}):
            output = run_command(self.launch_args + testargs, return_stdout=True)
            results = re.findall("({.+})", output)
            results = [r for r in results if "accuracy" in r][-1]
            results = ast.literal_eval(results)
            assert results["accuracy"] >= 0.75

    def test_multi_process_metrics(self):
        testargs = ["examples/by_feature/multi_process_metrics.py"]
        run_command(self.launch_args + testargs)

    @require_trackers
    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline", "DVCLIVE_TEST": "true"})
    def test_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            testargs = f"""
            examples/by_feature/tracking.py
            --with_tracking
            --project_dir {tmpdir}
            """.split()
            run_command(self.launch_args + testargs)
            assert os.path.exists(os.path.join(tmpdir, "tracking"))

    def test_gradient_accumulation(self):
        testargs = ["examples/by_feature/gradient_accumulation.py"]
        run_command(self.launch_args + testargs)

    def test_local_sgd(self):
        testargs = ["examples/by_feature/local_sgd.py"]
        run_command(self.launch_args + testargs)

    def test_early_stopping(self):
        testargs = ["examples/by_feature/early_stopping.py"]
        run_command(self.launch_args + testargs)

    @require_pippy
    @require_multi_gpu
    def test_pippy_examples_bert(self):
        testargs = ["examples/inference/bert.py"]
        run_command(self.launch_args + testargs)

    @require_pippy
    @require_multi_gpu
    def test_pippy_examples_gpt2(self):
        testargs = ["examples/inference/gpt2.py"]
        run_command(self.launch_args + testargs)

    @require_pippy
    @require_multi_gpu
    def test_pippy_examples_t5(self):
        testargs = ["examples/inference/t5.py"]
        run_command(self.launch_args + testargs)

    @slow
    @require_pippy
    @require_multi_gpu
    def test_pippy_examples_llama(self):
        testargs = ["examples/inference/llama.py"]
        run_command(self.launch_args + testargs)
