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

import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# We use TF to parse the logs
from accelerate import Accelerator
from accelerate.test_utils.testing import require_tensorflow
from accelerate.utils import is_tensorflow_available


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
            oldpwd = os.getcwd()
            os.chdir(dirpath)
            accelerator = Accelerator(log_with="tensorboard")
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(project_name, config)
            accelerator.end_training()
            for child in Path(project_name).glob("*/**"):
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
            os.chdir(oldpwd)

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
            oldpwd = os.getcwd()
            os.chdir(dirpath)
            accelerator = Accelerator(log_with="tensorboard")
            accelerator.init_trackers(project_name)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            # Logged values are stored in the outermost-tfevents file and can be read in as a TFRecord
            # Names are randomly generated each time
            log = list(filter(lambda x: x.is_file(), Path(project_name).iterdir()))[0]
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
            os.chdir(oldpwd)
        self.assertAlmostEqual(total_loss, values["total_loss"])
        self.assertEqual(iteration, values["iteration"])
        self.assertEqual(my_text, values["my_text"])


@mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
class WandBTrackingTest(unittest.TestCase):
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
        oldpwd = os.getcwd()
        with tempfile.TemporaryDirectory() as dirpath:
            os.chmod(dirpath, 0o777) # Is needed for wandb to write to it
            os.chdir(dirpath)
            accelerator = Accelerator(log_with="wandb")
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(project_name, config)
            accelerator.end_training()
            # The latest offline log is stored at wandb/latest-run/*.wandb
            for child in Path(f"{dirpath}/wandb/latest-run").glob("*"):
                logger.info(child)
                if child.is_file() and child.suffix == ".wandb":
                    with open(child, "rb") as f:
                        content = f.read()
                    break
        os.chdir(oldpwd)

        # Check HPS through careful parsing and cleaning
        cleaned_log = re.sub(r"[\x00-\x1f]+", " ", content.decode("utf8", "ignore"))
        self.assertEqual(self.get_value_from_log("num_iterations", cleaned_log), "12")
        self.assertEqual(self.get_value_from_log("learning_rate", cleaned_log), "0.01")
        self.assertEqual(self.get_value_from_log("some_boolean", cleaned_log), "false")
        self.assertEqual(self.get_value_from_log("some_string", cleaned_log), "some_value")

    def test_log(self):
        project_name = "test_project_with_log"
        oldpwd = os.getcwd()
        with tempfile.TemporaryDirectory() as dirpath:
            os.chdir(dirpath)
            accelerator = Accelerator(log_with="wandb")
            accelerator.init_trackers(project_name)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            # The latest offline log is stored at wandb/latest-run/*.wandb
            for child in Path("wandb").glob("*"):
                logger.debug(child)
            for child in Path("wandb/latest-run").glob("*"):
                logger.debug(child)
                if child.is_file() and child.suffix == ".wandb":
                    with open(child, "rb") as f:
                        content = f.read()
                    break
        os.chdir(oldpwd)
        # Check HPS through careful parsing and cleaning
        cleaned_log = re.sub(r"[\x00-\x1f]+", " ", content.decode("utf8", "ignore"))
        self.assertEqual(self.get_value_from_log("total_loss", cleaned_log), "0.1")
        self.assertEqual(self.get_value_from_log("iteration", cleaned_log), "1")
        self.assertEqual(self.get_value_from_log("my_text", cleaned_log), "some_value")
        self.assertEqual(self.get_value_from_log("_step", cleaned_log), "0")
