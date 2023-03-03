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

import csv
import json
import logging
import os
import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Optional
from unittest import mock

# We use TF to parse the logs
from accelerate import Accelerator
from accelerate.test_utils.testing import (
    MockingTestCase,
    TempDirTestCase,
    require_comet_ml,
    require_tensorboard,
    require_wandb,
    skip,
)
from accelerate.tracking import CometMLTracker, GeneralTracker
from accelerate.utils import is_comet_ml_available


if is_comet_ml_available():
    from comet_ml import OfflineExperiment

logger = logging.getLogger(__name__)


@require_tensorboard
class TensorBoardTrackingTest(unittest.TestCase):
    def test_init_trackers(self):
        project_name = "test_project_with_config"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", logging_dir=dirpath)
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(project_name, config)
            accelerator.end_training()
            for child in Path(f"{dirpath}/{project_name}").glob("*/**"):
                log = list(filter(lambda x: x.is_file(), child.iterdir()))[0]
            self.assertNotEqual(str(log), "")

    def test_log(self):
        project_name = "test_project_with_log"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", project_dir=dirpath)
            accelerator.init_trackers(project_name)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            # Logged values are stored in the outermost-tfevents file and can be read in as a TFRecord
            # Names are randomly generated each time
            log = list(filter(lambda x: x.is_file(), Path(f"{dirpath}/{project_name}").iterdir()))[0]
            self.assertNotEqual(str(log), "")

    def test_project_dir(self):
        with self.assertRaisesRegex(ValueError, "Logging with `tensorboard` requires a `logging_dir`"):
            _ = Accelerator(log_with="tensorboard")
        with tempfile.TemporaryDirectory() as dirpath:
            _ = Accelerator(log_with="tensorboard", project_dir=dirpath)
        with tempfile.TemporaryDirectory() as dirpath:
            _ = Accelerator(log_with="tensorboard", logging_dir=dirpath)


@require_wandb
@mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
class WandBTrackingTest(TempDirTestCase, MockingTestCase):
    def setUp(self):
        super().setUp()
        # wandb let's us override where logs are stored to via the WANDB_DIR env var
        self.add_mocks(mock.patch.dict(os.environ, {"WANDB_DIR": self.tmpdir}))

    @staticmethod
    def parse_log(log: str, section: str, record: bool = True):
        """
        Parses wandb log for `section` and returns a dictionary of
        all items in that section. Section names are based on the
        output of `wandb sync --view --verbose` and items starting
        with "Record" in that result
        """
        # Big thanks to the W&B team for helping us parse their logs
        pattern = rf"{section} ([\S\s]*?)\n\n"
        if record:
            pattern = rf"Record: {pattern}"
        cleaned_record = re.findall(pattern, log)[0]
        # A config
        if section == "config" or section == "history":
            cleaned_record = re.findall(r'"([a-zA-Z0-9_.,]+)', cleaned_record)
            return {key: val for key, val in zip(cleaned_record[0::2], cleaned_record[1::2])}
        # Everything else
        else:
            return dict(re.findall(r'(\w+): "([^\s]+)"', cleaned_record))

    @skip
    def test_wandb(self):
        project_name = "test_project_with_config"
        accelerator = Accelerator(log_with="wandb")
        config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
        kwargs = {"wandb": {"tags": ["my_tag"]}}
        accelerator.init_trackers(project_name, config, kwargs)
        values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
        accelerator.log(values, step=0)
        accelerator.end_training()
        # The latest offline log is stored at wandb/latest-run/*.wandb
        for child in Path(f"{self.tmpdir}/wandb/latest-run").glob("*"):
            if child.is_file() and child.suffix == ".wandb":
                content = subprocess.check_output(
                    ["wandb", "sync", "--view", "--verbose", str(child)], env=os.environ.copy()
                ).decode("utf8", "ignore")
                break

        # Check HPS through careful parsing and cleaning
        logged_items = self.parse_log(content, "config")
        self.assertEqual(logged_items["num_iterations"], "12")
        self.assertEqual(logged_items["learning_rate"], "0.01")
        self.assertEqual(logged_items["some_boolean"], "false")
        self.assertEqual(logged_items["some_string"], "some_value")
        self.assertEqual(logged_items["some_string"], "some_value")

        # Run tags
        logged_items = self.parse_log(content, "run", False)
        self.assertEqual(logged_items["tags"], "my_tag")

        # Actual logging
        logged_items = self.parse_log(content, "history")
        self.assertEqual(logged_items["total_loss"], "0.1")
        self.assertEqual(logged_items["iteration"], "1")
        self.assertEqual(logged_items["my_text"], "some_value")
        self.assertEqual(logged_items["_step"], "0")


# Comet has a special `OfflineExperiment` we need to use for testing
def offline_init(self, run_name: str, tmpdir: str):
    self.run_name = run_name
    self.writer = OfflineExperiment(project_name=run_name, offline_directory=tmpdir)
    logger.info(f"Initialized offline CometML project {self.run_name}")
    logger.info("Make sure to log any initial configurations with `self.store_init_configuration` before training!")


@require_comet_ml
@mock.patch.object(CometMLTracker, "__init__", offline_init)
class CometMLTest(unittest.TestCase):
    @staticmethod
    def get_value_from_key(log_list, key: str, is_param: bool = False):
        "Extracts `key` from Comet `log`"
        for log in log_list:
            j = json.loads(log)["payload"]
            if is_param and "param" in j.keys():
                if j["param"]["paramName"] == key:
                    return j["param"]["paramValue"]
            if "log_other" in j.keys():
                if j["log_other"]["key"] == key:
                    return j["log_other"]["val"]
            if "metric" in j.keys():
                if j["metric"]["metricName"] == key:
                    return j["metric"]["metricValue"]

    def test_init_trackers(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = CometMLTracker("test_project_with_config", d)
            accelerator = Accelerator(log_with=tracker)
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(None, config)
            accelerator.end_training()
            log = os.listdir(d)[0]  # Comet is nice, it's just a zip file here
            # We parse the raw logs
            p = os.path.join(d, log)
            archive = zipfile.ZipFile(p, "r")
            log = archive.open("messages.json").read().decode("utf-8")
        list_of_json = log.split("\n")[:-1]
        self.assertEqual(self.get_value_from_key(list_of_json, "num_iterations", True), 12)
        self.assertEqual(self.get_value_from_key(list_of_json, "learning_rate", True), 0.01)
        self.assertEqual(self.get_value_from_key(list_of_json, "some_boolean", True), False)
        self.assertEqual(self.get_value_from_key(list_of_json, "some_string", True), "some_value")

    def test_log(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = CometMLTracker("test_project_with_config", d)
            accelerator = Accelerator(log_with=tracker)
            accelerator.init_trackers(None)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            log = os.listdir(d)[0]  # Comet is nice, it's just a zip file here
            # We parse the raw logs
            p = os.path.join(d, log)
            archive = zipfile.ZipFile(p, "r")
            log = archive.open("messages.json").read().decode("utf-8")
        list_of_json = log.split("\n")[:-1]
        self.assertEqual(self.get_value_from_key(list_of_json, "curr_step", True), 0)
        self.assertEqual(self.get_value_from_key(list_of_json, "total_loss"), 0.1)
        self.assertEqual(self.get_value_from_key(list_of_json, "iteration"), 1)
        self.assertEqual(self.get_value_from_key(list_of_json, "my_text"), "some_value")


class MyCustomTracker(GeneralTracker):
    "Basic tracker that writes to a csv for testing"
    _col_names = [
        "total_loss",
        "iteration",
        "my_text",
        "learning_rate",
        "num_iterations",
        "some_boolean",
        "some_string",
    ]

    name = "my_custom_tracker"
    requires_logging_directory = False

    def __init__(self, dir: str):
        self.f = open(f"{dir}/log.csv", "w+")
        self.writer = csv.DictWriter(self.f, fieldnames=self._col_names)
        self.writer.writeheader()

    @property
    def tracker(self):
        return self.writer

    def store_init_configuration(self, values: dict):
        logger.info("Call init")
        self.writer.writerow(values)

    def log(self, values: dict, step: Optional[int]):
        logger.info("Call log")
        self.writer.writerow(values)

    def finish(self):
        self.f.close()


class CustomTrackerTestCase(unittest.TestCase):
    def test_init_trackers(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = MyCustomTracker(d)
            accelerator = Accelerator(log_with=tracker)
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers("Some name", config)
            accelerator.end_training()
            with open(f"{d}/log.csv", "r") as f:
                data = csv.DictReader(f)
                data = next(data)
                truth = {
                    "total_loss": "",
                    "iteration": "",
                    "my_text": "",
                    "learning_rate": "0.01",
                    "num_iterations": "12",
                    "some_boolean": "False",
                    "some_string": "some_value",
                }
                self.assertDictEqual(data, truth)

    def test_log(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = MyCustomTracker(d)
            accelerator = Accelerator(log_with=tracker)
            accelerator.init_trackers("Some name")
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            with open(f"{d}/log.csv", "r") as f:
                data = csv.DictReader(f)
                data = next(data)
                truth = {
                    "total_loss": "0.1",
                    "iteration": "1",
                    "my_text": "some_value",
                    "learning_rate": "",
                    "num_iterations": "",
                    "some_boolean": "",
                    "some_string": "",
                }
                self.assertDictEqual(data, truth)
