# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import os
import sys
import tempfile
import unittest
from unittest import mock

from torch.utils.data import DataLoader

from accelerate import DistributedType
from accelerate.test_utils.examples import compare_against_test
from datasets import load_dataset
from transformers import AutoTokenizer


SRC_DIRS = [os.path.abspath(os.path.join("examples", "by_feature"))]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import checkpointing
    import tracking

# DataLoaders built from `test_samples/MRPC` for quick testing
# Should mock `{script_name}.get_dataloaders` via:
# @mock.patch("{script_name}.get_dataloaders", mocked_dataloaders)


def mocked_dataloaders(accelerator, batch_size: int = 16):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    data_files = {"train": "tests/test_samples/MRPC/train.csv", "validation": "tests/test_samples/MRPC/dev.csv"}
    datasets = load_dataset("csv", data_files=data_files)
    label_list = datasets["train"].unique("label")

    label_to_id = {v: i for i, v in enumerate(label_list)}

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"], examples["sentence2"], truncation=True, max_length=None, padding="max_length"
        )
        if "label" in examples:
            outputs["labels"] = [label_to_id[l] for l in examples["label"]]
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence1", "sentence2", "label"],
    )

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=2)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=1)

    return train_dataloader, eval_dataloader


class ExampleDifferenceTests(unittest.TestCase):
    """
    This TestCase checks that all of the `complete_*` scripts contain all of the
    information found in the `by_feature` scripts, line for line.
    """

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
        by_feature_path = os.path.abspath(os.path.join("examples", "by_feature"))
        examples_path = os.path.abspath("examples")
        for item in os.listdir(by_feature_path):
            item_path = os.path.join(by_feature_path, item)
            if os.path.isfile(item_path) and ".py" in item_path:
                with self.subTest(
                    tested_script=complete_file_name,
                    feature_script=item,
                    tested_section="main()" if parser_only else "training_function()",
                ):
                    diff = compare_against_test(
                        os.path.join(examples_path, complete_file_name), item_path, parser_only, secondary_filename
                    )
                    diff = "\n".join(diff)
                    if special_strings is not None:
                        for string in special_strings:
                            diff = diff.replace(string, "")
                    self.assertEqual(diff, "")

    def test_nlp_examples(self):
        self.one_complete_example("complete_nlp_example.py", True)
        self.one_complete_example("complete_nlp_example.py", False)

    def test_cv_examples(self):
        cv_path = os.path.abspath(os.path.join("examples", "cv_example.py"))
        special_strings = [
            " " * 16 + "{\n\n",
            " " * 18 + '"accuracy": eval_metric["accuracy"],\n\n',
            " " * 18 + '"f1": eval_metric["f1"],\n\n',
            " " * 18 + '"train_loss": total_loss,\n\n',
            " " * 18 + '"epoch": epoch,\n\n',
            " " * 16 + "}\n",
            " " * 8,
        ]
        self.one_complete_example("complete_cv_example.py", True, cv_path, special_strings)
        self.one_complete_example("complete_cv_example.py", False, cv_path, special_strings)


class FeatureExamplesTests(unittest.TestCase):
    @mock.patch("checkpointing.get_dataloaders", mocked_dataloaders)
    def test_checkpointing_by_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            testargs = f"""
            checkpointing.py
            --checkpointing_steps epoch
            --output_dir {tmpdir}
            """.split()
            with mock.patch.object(sys, "argv", testargs):
                checkpointing.main()
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "epoch_0")))

    @mock.patch("checkpointing.get_dataloaders", mocked_dataloaders)
    def test_checkpointing_by_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            testargs = f"""
            checkpointing.py
            --checkpointing_steps 2
            --output_dir {tmpdir}
            """.split()
            with mock.patch.object(sys, "argv", testargs):
                checkpointing.main()
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "step_2")))

    @mock.patch("tracking.get_dataloaders", mocked_dataloaders)
    def test_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_modules = dict(sys.modules)
            sys.modules.pop("comet_ml", None)
            sys.modules.pop("wandb", None)

            testargs = f"""
            tracking.py
            --with_tracking
            --logging_dir {tmpdir}
            """.split()
            with mock.patch.object(sys, "argv", testargs):
                tracking.main()
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "tracking")))
            sys.modules = old_modules
