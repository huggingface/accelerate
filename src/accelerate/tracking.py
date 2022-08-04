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

import os
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List, Optional, Union

from .logging import get_logger
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


logger = get_logger(__name__)


def get_available_trackers():
    "Returns a list of all supported available trackers in the system"
    return _available_trackers


class GeneralTracker(object, metaclass=ABCMeta):
    """
    A base Tracker class to be used for all logging integration implementations.

    Each function should take in `**kwargs` that will automatically be passed in from a base dictionary provided to
    [`Accelerator`]
    """

    @abstractproperty
    def name(self):
        "String representation of the python class name"
        pass

    @abstractproperty
    def requires_logging_directory(self):
        """
        Whether the logger requires a directory to store their logs. Should either return `True` or `False`.
        """
        pass

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
    def log(self, values: dict, step: Optional[int], **kwargs):
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

    def finish(self):
        """
        Should run any finalizing functions within the tracking API. If the API should not have one, just don't
        overwrite that method.
        """
        pass

    @abstractproperty
    def tracker(self):
        """
        Should return internal tracking mechanism used by a tracker class (such as the `run` for wandb)
        """
        pass


class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    def __init__(self, run_name: str, logging_dir: Optional[Union[str, os.PathLike]], **kwargs):
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = tensorboard.SummaryWriter(self.logging_dir, **kwargs)
        logger.info(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

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

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
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
        kwargs:
            Additional key word arguments passed along to the `wandb.init` method.
    """

    name = "wandb"
    requires_logging_directory = False

    def __init__(self, run_name: str, **kwargs):
        self.run_name = run_name
        self.run = wandb.init(project=self.run_name, **kwargs)
        logger.info(f"Initialized WandB project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.run

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

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        self.run.log(values, step=step, **kwargs)
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
        kwargs:
            Additional key word arguments passed along to the `Experiment.__init__` method.
    """

    name = "comet_ml"
    requires_logging_directory = False

    def __init__(self, run_name: str, **kwargs):
        self.run_name = run_name
        self.writer = Experiment(project_name=run_name, **kwargs)
        logger.info(f"Initialized CometML project {self.run_name}")
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

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

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `Experiment.log_metric`, `Experiment.log_other`,
                or `Experiment.log_metrics` method based on the contents of `values`.
        """
        if step is not None:
            self.writer.set_step(step)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.log_metric(k, v, step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.log_other(k, v, **kwargs)
            elif isinstance(v, dict):
                self.writer.log_metrics(v, step=step, **kwargs)
        logger.info("Successfully logged to CometML")

    def finish(self):
        """
        Closes `comet-ml` writer
        """
        self.writer.end()
        logger.info("CometML run closed")


LOGGER_TYPE_TO_CLASS = {"tensorboard": TensorBoardTracker, "wandb": WandBTracker, "comet_ml": CometMLTracker}


def filter_trackers(
    log_with: List[Union[str, LoggerType, GeneralTracker]], logging_dir: Union[str, os.PathLike] = None
):
    """
    Takes in a list of potential tracker types and checks that:
        - The tracker wanted is available in that environment
        - Filters out repeats of tracker types
        - If `all` is in `log_with`, will return all trackers in the environment
        - If a tracker requires a `logging_dir`, ensures that `logging_dir` is not `None`

    Args:
        log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
            A list of loggers to be setup for experiment tracking. Should be one or several of:

            - `"all"`
            - `"tensorboard"`
            - `"wandb"`
            - `"comet_ml"`
            If `"all`" is selected, will pick up all available trackers in the environment and intialize them. Can also
            accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        logging_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing logs of locally-compatible loggers.
    """
    loggers = []
    if log_with is not None:
        if not isinstance(log_with, (list, tuple)):
            log_with = [log_with]
            logger.debug(f"{log_with}")
        if "all" in log_with or LoggerType.ALL in log_with:
            loggers = [o for o in log_with if issubclass(type(o), GeneralTracker)] + get_available_trackers()
        else:
            for log_type in log_with:
                if log_type not in LoggerType and not issubclass(type(log_type), GeneralTracker):
                    raise ValueError(f"Unsupported logging capability: {log_type}. Choose between {LoggerType.list()}")
                if issubclass(type(log_type), GeneralTracker):
                    loggers.append(log_type)
                else:
                    log_type = LoggerType(log_type)
                    if log_type not in loggers:
                        if log_type in get_available_trackers():
                            tracker_init = LOGGER_TYPE_TO_CLASS[str(log_type)]
                            if getattr(tracker_init, "requires_logging_directory"):
                                if logging_dir is None:
                                    raise ValueError(
                                        f"Logging with `{str(log_type)}` requires a `logging_dir` to be passed in."
                                    )
                            loggers.append(log_type)
                        else:
                            logger.info(f"Tried adding logger {log_type}, but package is unavailable in the system.")

    return loggers
