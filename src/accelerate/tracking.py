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

# Expectation:
# Provide a project dir name, then each type of logger gets stored in project/{`logging_dir`}

import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

from .utils import LoggerType, is_tensorboard_available


if is_tensorboard_available():
    from torch.utils import tensorboard


logger = logging.getLogger(__name__)


def get_available_trackers():
    trackers = []
    if is_tensorboard_available:
        trackers.append(LoggerType.TENSORBOARD)
    return trackers


class GeneralTracker(object, metaclass=ABCMeta):
    """
    A base Tracker class to be used for all logging integration implementations.
    """

    log_directory: str

    @abstractmethod
    def store_init_configuration(self, values):
        pass

    @abstractmethod
    def log(self, values):
        pass


class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. 
    Should be initialized at the start of your script. 

    Args:
        run_name (`str`):
            The name of the experiment run. Logs are then stored in `tensorboard/{run_name}`
    """
    log_directory = Path("tensorboard")

    def __init__(self, run_name:str=""):
        self.run_name = run_name
        self.writer = tensorboard.SummaryWriter(self.logging_dir/self.run_name)
        logger.info(f"Initialized TensorBoard project {self.run_name} writing to {self.logging_dir}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values:dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs. Value be of type `bool`, `str`, `float`, `int`, or `None`.
        """
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        logger.info("Stored initial configuration hyperparameters to TensorBoard")

    # add_scalar has the option for `global_step`, should we include this as an optional argument?
    # Need to see how rest of API fleshes out first with other integrations
    # Potential other option: keep track of number of times `optimizer.step` is called internally
    def log(self, values: dict):
        """
        Logs `values` to the current run.
        If `global_step` is included as a key, this will be logged directly in the internal call.
        
        Args:
            values (`dict`):
                Values to be logged as key-value pairs. Value be of type `int`, `float`, or `str`.
        """
        global_step = None
        if "global_step" in values.keys():
            global_step = values["global_step"]
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=global_step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=global_step)
        self.writer.flush()
        logger.info("Successfully logged to TensorBoard")

# class WandBTracker(GeneralTracker):
#     """
#     """
