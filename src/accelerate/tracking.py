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
from typing import Optional

from .utils import LoggerType, is_comet_ml_available, is_tensorboard_available, is_wandb_available


_available_trackers = []

if is_tensorboard_available():
    from torch.utils import tensorboard

    _available_trackers.append(LoggerType.TENSORBOARD)

if is_wandb_available():
    import wandb

    _available_trackers.append(LoggerType.WANDB)

if is_comet_ml_available():
    from comet_ml import Experiment

    _available_trackers.append(LoggerType.COMETML)


logger = logging.getLogger(__name__)


def get_available_trackers():
    "Returns a list of all supported available trackers in the system"
    return _available_trackers


class GeneralTracker(object, metaclass=ABCMeta):
    """
    A base Tracker class to be used for all logging integration implementations.
    """

    @abstractmethod
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Implementations should use the experiment configuration
        functionality of a tracking API.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        pass

    @abstractmethod
    def log(self, values: dict, step: Optional[int]):
        """
        Logs `values` to the current run. Base `log` implementations of a tracking API should go in here, along with
        special behavior for the `step parameter.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        pass

    @abstractmethod
    def finish(self):
        """
        Should run any finalizing functions within the tracking API. If the API should not have one, just return:
        ```python
        super().finish()
        ```
        """
        pass


class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
    """

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.writer = tensorboard.SummaryWriter(self.run_name)
        logger.info(f"Initialized TensorBoard project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        logger.info("Stored initial configuration hyperparameters to TensorBoard")

    def log(self, values: dict, step: Optional[int] = None):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step)
        self.writer.flush()
        logger.info("Successfully logged to TensorBoard")

    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.info("TensorBoard writer closed")


class WandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
    """

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.run = wandb.init(self.run_name)
        logger.info(f"Initialized WandB project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        wandb.config.update(values)
        logger.info("Stored initial configuration hyperparameters to WandB")

    def log(self, values: dict, step: Optional[int] = None):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        self.run.log(values, step=step)
        logger.info("Successfully logged to WandB")

    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.info("WandB run closed")


class CometMLTracker(GeneralTracker):
    """
    A `Tracker` class that supports `comet_ml`. Should be initialized at the start of your script.

    API keys must be stored in a Comet config file.

    Args:
        run_name (`str`):
            The name of the experiment run.
    """

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.writer = Experiment(project_name=run_name)
        logger.info(f"Initialized CometML project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.log_parameters(values)
        logger.info("Stored initial configuration hyperparameters to CometML")

    def log(self, values: dict, step: Optional[int] = None):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        if step is not None:
            self.writer.set_step(step)
        self.writer.log_others(values)
        logger.info("Successfully logged to CometML")

    def finish(self):
        """Do nothing"""
        super().finish()
