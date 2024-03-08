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

import numpy as np
import torch

# We use TF to parse the logs
from accelerate import Accelerator
from accelerate.test_utils.testing import (
    MockingTestCase,
    TempDirTestCase,
    require_clearml,
    require_comet_ml,
    require_dvclive,
    require_pandas,
    require_tensorboard,
    require_wandb,
    skip,
)
from accelerate.tracking import CometMLTracker, GeneralTracker
from accelerate.utils import (
    ProjectConfiguration,
    is_comet_ml_available,
    is_dvclive_available,
    is_tensorboard_available,
)


if is_comet_ml_available():
    from comet_ml import OfflineExperiment

if is_tensorboard_available():
    import struct

    import tensorboard.compat.proto.event_pb2 as event_pb2

if is_dvclive_available():
    from dvclive.plots.metric import Metric
    from dvclive.serialize import load_yaml
    from dvclive.utils import parse_metrics

logger = logging.getLogger(__name__)


@require_tensorboard
class TensorBoardTrackingTest(unittest.TestCase):
    def test_init_trackers(self):
        project_name = "test_project_with_config"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", project_dir=dirpath)
            config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
            accelerator.init_trackers(project_name, config)
            accelerator.end_training()
            for child in Path(f"{dirpath}/{project_name}").glob("*/**"):
                log = list(filter(lambda x: x.is_file(), child.iterdir()))[0]
            assert str(log) != ""

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
            assert str(log) != ""

    def test_log_with_tensor(self):
        project_name = "test_project_with_log"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="tensorboard", project_dir=dirpath)
            accelerator.init_trackers(project_name)
            values = {"tensor": torch.tensor(1)}
            accelerator.log(values, step=0)
            accelerator.end_training()
            # Logged values are stored in the outermost-tfevents file and can be read in as a TFRecord
            # Names are randomly generated each time
            log = list(filter(lambda x: x.is_file(), Path(f"{dirpath}/{project_name}").iterdir()))[0]
            # Reading implementation based on https://github.com/pytorch/pytorch/issues/45327#issuecomment-703757685
            with open(log, "rb") as f:
                data = f.read()
            found_tensor = False
            while data:
                header = struct.unpack("Q", data[:8])

                event_str = data[12 : 12 + int(header[0])]  # 8+4
                data = data[12 + int(header[0]) + 4 :]
                event = event_pb2.Event()

                event.ParseFromString(event_str)
                if event.HasField("summary"):
                    for value in event.summary.value:
                        if value.simple_value == 1.0 and value.tag == "tensor":
                            found_tensor = True
            assert found_tensor, "Converted tensor was not found in the log file!"

    def test_project_dir(self):
        with self.assertRaisesRegex(ValueError, "Logging with `tensorboard` requires a `logging_dir`"):
            _ = Accelerator(log_with="tensorboard")
        with tempfile.TemporaryDirectory() as dirpath:
            _ = Accelerator(log_with="tensorboard", project_dir=dirpath)

    def test_project_dir_with_config(self):
        config = ProjectConfiguration(total_limit=30)
        with tempfile.TemporaryDirectory() as dirpath:
            _ = Accelerator(log_with="tensorboard", project_dir=dirpath, project_config=config)


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
                cmd = ["wandb", "sync", "--view", "--verbose", str(child)]
                content = subprocess.check_output(cmd, encoding="utf8", errors="ignore")
                break

        # Check HPS through careful parsing and cleaning
        logged_items = self.parse_log(content, "config")
        assert logged_items["num_iterations"] == "12"
        assert logged_items["learning_rate"] == "0.01"
        assert logged_items["some_boolean"] == "false"
        assert logged_items["some_string"] == "some_value"
        assert logged_items["some_string"] == "some_value"

        # Run tags
        logged_items = self.parse_log(content, "run", False)
        assert logged_items["tags"] == "my_tag"

        # Actual logging
        logged_items = self.parse_log(content, "history")
        assert logged_items["total_loss"] == "0.1"
        assert logged_items["iteration"] == "1"
        assert logged_items["my_text"] == "some_value"
        assert logged_items["_step"] == "0"


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
        assert self.get_value_from_key(list_of_json, "num_iterations", True) == 12
        assert self.get_value_from_key(list_of_json, "learning_rate", True) == 0.01
        assert self.get_value_from_key(list_of_json, "some_boolean", True) is False
        assert self.get_value_from_key(list_of_json, "some_string", True) == "some_value"

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
        assert self.get_value_from_key(list_of_json, "curr_step", True) == 0
        assert self.get_value_from_key(list_of_json, "total_loss") == 0.1
        assert self.get_value_from_key(list_of_json, "iteration") == 1
        assert self.get_value_from_key(list_of_json, "my_text") == "some_value"


@require_clearml
class ClearMLTest(TempDirTestCase, MockingTestCase):
    def setUp(self):
        super().setUp()
        # ClearML offline session location is stored in CLEARML_CACHE_DIR
        self.add_mocks(mock.patch.dict(os.environ, {"CLEARML_CACHE_DIR": self.tmpdir}))

    @staticmethod
    def _get_offline_dir(accelerator):
        from clearml.config import get_offline_dir

        return get_offline_dir(task_id=accelerator.get_tracker("clearml", unwrap=True).id)

    @staticmethod
    def _get_metrics(offline_dir):
        metrics = []
        with open(os.path.join(offline_dir, "metrics.jsonl")) as f:
            json_lines = f.readlines()
            for json_line in json_lines:
                metrics.extend(json.loads(json_line))
        return metrics

    def test_init_trackers(self):
        from clearml import Task
        from clearml.utilities.config import text_to_config_dict

        Task.set_offline(True)
        accelerator = Accelerator(log_with="clearml")
        config = {"num_iterations": 12, "learning_rate": 1e-2, "some_boolean": False, "some_string": "some_value"}
        accelerator.init_trackers("test_project_with_config", config)

        offline_dir = ClearMLTest._get_offline_dir(accelerator)
        accelerator.end_training()

        with open(os.path.join(offline_dir, "task.json")) as f:
            offline_session = json.load(f)
        clearml_offline_config = text_to_config_dict(offline_session["configuration"]["General"]["value"])
        assert config == clearml_offline_config

    def test_log(self):
        from clearml import Task

        Task.set_offline(True)
        accelerator = Accelerator(log_with="clearml")
        accelerator.init_trackers("test_project_with_log")
        values_with_iteration = {"should_be_under_train": 1, "eval_value": 2, "test_value": 3.1, "train_value": 4.1}
        accelerator.log(values_with_iteration, step=1)
        single_values = {"single_value_1": 1.1, "single_value_2": 2.2}
        accelerator.log(single_values)

        offline_dir = ClearMLTest._get_offline_dir(accelerator)
        accelerator.end_training()

        metrics = ClearMLTest._get_metrics(offline_dir)
        assert (len(values_with_iteration) + len(single_values)) == len(metrics)
        for metric in metrics:
            if metric["metric"] == "Summary":
                assert metric["variant"] in single_values
                assert metric["value"] == single_values[metric["variant"]]
            elif metric["metric"] == "should_be_under_train":
                assert metric["variant"] == "train"
                assert metric["iter"] == 1
                assert metric["value"] == values_with_iteration["should_be_under_train"]
            else:
                values_with_iteration_key = metric["variant"] + "_" + metric["metric"]
                assert values_with_iteration_key in values_with_iteration
                assert metric["iter"] == 1
                assert metric["value"] == values_with_iteration[values_with_iteration_key]

    def test_log_images(self):
        from clearml import Task

        Task.set_offline(True)
        accelerator = Accelerator(log_with="clearml")
        accelerator.init_trackers("test_project_with_log_images")

        base_image = np.eye(256, 256, dtype=np.uint8) * 255
        base_image_3d = np.concatenate((np.atleast_3d(base_image), np.zeros((256, 256, 2), dtype=np.uint8)), axis=2)
        images = {
            "base_image": base_image,
            "base_image_3d": base_image_3d,
        }
        accelerator.get_tracker("clearml").log_images(images, step=1)

        offline_dir = ClearMLTest._get_offline_dir(accelerator)
        accelerator.end_training()

        images_saved = Path(os.path.join(offline_dir, "data")).rglob("*.jpeg")
        assert len(list(images_saved)) == len(images)

    def test_log_table(self):
        from clearml import Task

        Task.set_offline(True)
        accelerator = Accelerator(log_with="clearml")
        accelerator.init_trackers("test_project_with_log_table")

        accelerator.get_tracker("clearml").log_table(
            "from lists with columns", columns=["A", "B", "C"], data=[[1, 3, 5], [2, 4, 6]]
        )
        accelerator.get_tracker("clearml").log_table("from lists", data=[["A2", "B2", "C2"], [7, 9, 11], [8, 10, 12]])
        offline_dir = ClearMLTest._get_offline_dir(accelerator)
        accelerator.end_training()

        metrics = ClearMLTest._get_metrics(offline_dir)
        assert len(metrics) == 2
        for metric in metrics:
            assert metric["metric"] in ("from lists", "from lists with columns")
            plot = json.loads(metric["plot_str"])
            if metric["metric"] == "from lists with columns":
                print(plot["data"][0])
                self.assertCountEqual(plot["data"][0]["header"]["values"], ["A", "B", "C"])
                self.assertCountEqual(plot["data"][0]["cells"]["values"], [[1, 2], [3, 4], [5, 6]])
            else:
                self.assertCountEqual(plot["data"][0]["header"]["values"], ["A2", "B2", "C2"])
                self.assertCountEqual(plot["data"][0]["cells"]["values"], [[7, 8], [9, 10], [11, 12]])

    @require_pandas
    def test_log_table_pandas(self):
        import pandas as pd
        from clearml import Task

        Task.set_offline(True)
        accelerator = Accelerator(log_with="clearml")
        accelerator.init_trackers("test_project_with_log_table_pandas")

        accelerator.get_tracker("clearml").log_table(
            "from df", dataframe=pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]}), step=1
        )

        offline_dir = ClearMLTest._get_offline_dir(accelerator)
        accelerator.end_training()

        metrics = ClearMLTest._get_metrics(offline_dir)
        assert len(metrics) == 1
        assert metrics[0]["metric"] == "from df"
        plot = json.loads(metrics[0]["plot_str"])
        self.assertCountEqual(plot["data"][0]["header"]["values"], [["A"], ["B"], ["C"]])
        self.assertCountEqual(plot["data"][0]["cells"]["values"], [[1, 2], [3, 4], [5, 6]])


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
            with open(f"{d}/log.csv") as f:
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
                assert data == truth

    def test_log(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = MyCustomTracker(d)
            accelerator = Accelerator(log_with=tracker)
            accelerator.init_trackers("Some name")
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            accelerator.log(values, step=0)
            accelerator.end_training()
            with open(f"{d}/log.csv") as f:
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
                assert data == truth


@require_dvclive
@mock.patch("dvclive.live.get_dvc_repo", return_value=None)
class DVCLiveTrackingTest(unittest.TestCase):
    def test_init_trackers(self, mock_repo):
        project_name = "test_project_with_config"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="dvclive")
            config = {
                "num_iterations": 12,
                "learning_rate": 1e-2,
                "some_boolean": False,
                "some_string": "some_value",
            }
            init_kwargs = {"dvclive": {"dir": dirpath, "save_dvc_exp": False, "dvcyaml": None}}
            accelerator.init_trackers(project_name, config, init_kwargs)
            accelerator.end_training()
            live = accelerator.trackers[0].live
            params = load_yaml(live.params_file)
            assert params == config

    def test_log(self, mock_repo):
        project_name = "test_project_with_log"
        with tempfile.TemporaryDirectory() as dirpath:
            accelerator = Accelerator(log_with="dvclive", project_dir=dirpath)
            init_kwargs = {"dvclive": {"dir": dirpath, "save_dvc_exp": False, "dvcyaml": None}}
            accelerator.init_trackers(project_name, init_kwargs=init_kwargs)
            values = {"total_loss": 0.1, "iteration": 1, "my_text": "some_value"}
            # Log step 0
            accelerator.log(values)
            # Log step 1
            accelerator.log(values)
            # Log step 3 (skip step 2)
            accelerator.log(values, step=3)
            accelerator.end_training()
            live = accelerator.trackers[0].live
            logs, latest = parse_metrics(live)
            assert latest.pop("step") == 3
            assert latest == values
            scalars = os.path.join(live.plots_dir, Metric.subfolder)
            for val in values.keys():
                val_path = os.path.join(scalars, f"{val}.tsv")
                steps = [int(row["step"]) for row in logs[val_path]]
                assert steps == [0, 1, 3]
