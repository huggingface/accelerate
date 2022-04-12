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

    def one_complete_example(self, complete_file_name: str):
        """
        Tests a single `complete` example against all of the implemented `by_feature` scripts

        Args:
            complete_file_name (`str`):
                The filename of a complete example
        """
        by_feature_path = os.path.abspath(os.path.join("examples", "by_feature"))
        examples_path = os.path.abspath("examples")
        for item in os.listdir(by_feature_path):
            item_path = os.path.join(by_feature_path, item)
            if os.path.isfile(item_path) and ".py" in item_path:
                with self.subTest(feature_script=item):
                    diff = compare_against_test(
                        os.path.join(examples_path, "nlp_example.py"),
                        os.path.join(examples_path, complete_file_name),
                        item_path,
                    )
                    self.assertEqual(diff, [])

    def test_complete_nlp_example(self):
        self.one_complete_example("complete_nlp_example.py")

    def test_complete_cv_example(self):
        self.one_complete_example("complete_cv_example.py")


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
        testargs = ["tracking.py"]
        with mock.patch.object(sys, "argv", testargs):
            tracking.main()
