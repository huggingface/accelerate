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
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Optional
from unittest import mock

# We use TF to parse the logs
from accelerate import Accelerator
from accelerate.test_utils.testing import MockingTestCase, TempDirTestCase, require_tensorflow
from accelerate.tracking import CometMLTracker, GeneralTracker
from accelerate.utils import is_tensorflow_available
from comet_ml import OfflineExperiment


if is_tensorflow_available():
    import tensorflow as tf
    from tensorboard.plugins.hparams import plugin_data_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.python.summary.summary_iterator import summary_iterator


logger = logging.getLogger(__name__)


class TensorBoardTrackingTest(unittest.TestCase):
    @require_tensorflow
    def test_init_trackers(self):
        hps = None
        project_name = "test_project_with_config"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", logging_dir=dirpath)
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(project_name, config)
            accelerator.end_training()
            for child in Path(f"{dirpath}/{project_name}").glob("*/**"):
                log = list(filter(lambda x: x.is_file(), child.iterdir()))[0]
                # The config log is stored one layer deeper in the logged directory
                # And names are randomly generated each time
            si = summary_iterator(str(log))
            # Pull HPS through careful parsing
            for event in si:
                for value in event.summary.value:
                    proto_bytes = value.metadata.plugin_data.content
                    plugin_data = plugin_data_pb2.HParamsPluginData.FromString(proto_bytes)
                    if plugin_data.HasField("session_start_info"):
                        hps = dict(plugin_data.session_start_info.hparams)

        self.assertTrue(isinstance(hps, dict))
        keys = list(hps.keys())
        keys.sort()
        self.assertEqual(keys, ["learning_rate", "num_iterations", "some_boolean", "some_string"])
        self.assertEqual(hps["num_iterations"].number_value, 12)
        self.assertEqual(hps["learning_rate"].number_value, 0.01)
        self.assertEqual(hps["some_boolean"].bool_value, False)
        self.assertEqual(hps["some_string"].string_value, "some_value")

    @require_tensorflow
    def test_log(self):
        step = None
        project_name = "test_project_with_log"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", logging_dir=dirpath)
            accelerator.init_trackers(project_name)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            # Logged values are stored in the outermost-tfevents file and can be read in as a TFRecord
            # Names are randomly generated each time
            log = list(filter(lambda x: x.is_file(), Path(f"{dirpath}/{project_name}").iterdir()))[0]
            serialized_examples = tf.data.TFRecordDataset(log)
            for e in serialized_examples:
                event = event_pb2.Event.FromString(e.numpy())
                if step is None:
                    step = event.step
                for value in event.summary.value:
                    if value.tag == "total_loss":
                        total_loss = value.simple_value
                    elif value.tag == "iteration":
                        iteration = value.simple_value
                    elif value.tag == "my_text/text_summary":  # Append /text_summary to the key
                        my_text = value.tensor.string_val[0].decode()
        self.assertAlmostEqual(total_loss, values["total_loss"])
        self.assertEqual(iteration, values["iteration"])
        self.assertEqual(my_text, values["my_text"])

    def test_logging_dir(self):
        with self.assertRaisesRegex(ValueError, "Logging with `tensorboard` requires a `logging_dir`"):
            _ = Accelerator(log_with="tensorboard")
        with tempfile.TemporaryDirectory() as dirpath:
            _ = Accelerator(log_with="tensorboard", logging_dir=dirpath)


@mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
class WandBTrackingTest(TempDirTestCase, MockingTestCase):
    def setUp(self):
        super().setUp()
        # wandb let's us override where logs are stored to via the WANDB_DIR env var
        self.add_mocks(mock.patch.dict(os.environ, {"WANDB_DIR": self.tmpdir}))

    @staticmethod
    def get_value_from_log(key: str, log: str, key_occurance: int = 0):
        """
        Parses wandb log for `key` and returns the value.
        If parsing through multiple calls to .log, pass in a `key_occurance`
        """
        res = re.findall(rf"(?<={key} )[^\s]+", log)[key_occurance]
        if '"' in res:
            return re.findall(r'"([^"]*)"', res)[0]
        else:
            return res

    def test_init_trackers(self):
        project_name = "test_project_with_config"
        accelerator = Accelerator(log_with="wandb")
        config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
        accelerator.init_trackers(project_name, config)
        accelerator.end_training()
        # The latest offline log is stored at wandb/latest-run/*.wandb
        for child in Path(f"{self.tmpdir}/wandb/latest-run").glob("*"):
            logger.info(child)
            if child.is_file() and child.suffix == ".wandb":
                with open(child, "rb") as f:
                    content = f.read()
                break

        # Check HPS through careful parsing and cleaning
        cleaned_log = re.sub(r"[\x00-\x1f]+", " ", content.decode("utf8", "ignore"))
        self.assertEqual(self.get_value_from_log("num_iterations", cleaned_log), "12")
        self.assertEqual(self.get_value_from_log("learning_rate", cleaned_log), "0.01")
        self.assertEqual(self.get_value_from_log("some_boolean", cleaned_log), "false")
        self.assertEqual(self.get_value_from_log("some_string", cleaned_log), "some_value")

    def test_log(self):
        project_name = "test_project_with_log"
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(project_name)
        values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
        accelerator.log(values, step=0)
        accelerator.end_training()
        # The latest offline log is stored at wandb/latest-run/*.wandb
        for child in Path(f"{self.tmpdir}/wandb/latest-run").glob("*"):
            if child.is_file() and child.suffix == ".wandb":
                with open(child, "rb") as f:
                    content = f.read()
                break
        # Check HPS through careful parsing and cleaning
        cleaned_log = re.sub(r"[\x00-\x1f]+", " ", content.decode("utf8", "ignore"))
        self.assertTrue("0.1" in self.get_value_from_log("total_loss", cleaned_log))
        self.assertTrue("1" in self.get_value_from_log("iteration", cleaned_log))
        self.assertTrue("some_value" in self.get_value_from_log("my_text", cleaned_log))
        self.assertTrue("0" in self.get_value_from_log("_step", cleaned_log))


# Comet has a special `OfflineExperiment` we need to use for testing
def offline_init(self, run_name: str, tmpdir: str):
    self.run_name = run_name
    self.writer = OfflineExperiment(project_name=run_name, offline_directory=tmpdir)
    logger.info(f"Initialized offline CometML project {self.run_name}")
    logger.info("Make sure to log any initial configurations with `self.store_init_configuration` before training!")


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

    requires_logging_directory = False

    def __init__(self, dir: str):
        self.f = open(f"{dir}/log.csv", "w+")
        self.writer = csv.DictWriter(self.f, fieldnames=self._col_names)
        self.writer.writeheader()

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
