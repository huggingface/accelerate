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
from .utils import is_tensorboard_available

if is_tensorboard_available():
    from torch.utils import tensorboard


logger = logging.getLogger(__name__)


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
    log_directory = "tensorboard"

    def __init__(self, run_name=""):
        self.run_name = Path(run_name)
        self.writer = tensorboard.SummaryWriter(self.logging_dir)
        logger.info(f"Initialized TensorBoard project {self.run_name} writing to {self.logging_dir}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values):
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        logger.info("Stored initial configuration hyperparameters to TensorBoard")

    # add_scalar has the option for `global_step`, should we include this?
    # Need to see how rest of API fleshes out first with other integrations
    def log(self, values: dict):
        """
        Logs `values`. Values should be a dictionary-like object containing only types `int`, `float`, or `str`.
        """
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v)
            elif isinstance(v, str):
                self.writer.add_text(k, v)
        self.writer.flush()
        logger.info("Successfully logged to TensorBoard")
