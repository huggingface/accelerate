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

import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import yaml

from .logging import get_logger
from .state import PartialState
from .utils import (
    LoggerType,
    is_aim_available,
    is_comet_ml_available,
    is_mlflow_available,
    is_tensorboard_available,
    is_wandb_available,
)


_available_trackers = []

if is_tensorboard_available():
    try:
        from torch.utils import tensorboard
    except ModuleNotFoundError:
        import tensorboardX as tensorboard

    _available_trackers.append(LoggerType.TENSORBOARD)

if is_wandb_available():
    import wandb

    _available_trackers.append(LoggerType.WANDB)

if is_comet_ml_available():
    from comet_ml import Experiment

    _available_trackers.append(LoggerType.COMETML)

if is_aim_available():
    from aim import Run

    _available_trackers.append(LoggerType.AIM)

if is_mlflow_available():
    import mlflow

    _available_trackers.append(LoggerType.MLFLOW)

logger = get_logger(__name__)


def on_main_process(function):
    """
    Decorator to selectively run the decorated function on the main process only based on the `main_process_only`
    attribute in a class.

    Checks at function execution rather than initialization time, not triggering the initialization of the
    `PartialState`.
    """

    @wraps(function)
    def execute_on_main_process(self, *args, **kwargs):
        if getattr(self, "main_process_only", False):
            return PartialState().on_main_process(function)(self, *args, **kwargs)
        else:
            return function(self, *args, **kwargs)

    return execute_on_main_process


def get_available_trackers():
    "Returns a list of all supported available trackers in the system"
    return _available_trackers


class GeneralTracker:
    """
    A base Tracker class to be used for all logging integration implementations.

    Each function should take in `**kwargs` that will automatically be passed in from a base dictionary provided to
    [`Accelerator`].

    Should implement `name`, `requires_logging_directory`, and `tracker` properties such that:

    `name` (`str`): String representation of the tracker class name, such as "TensorBoard" `requires_logging_directory`
    (`bool`): Whether the logger requires a directory to store their logs. `tracker` (`object`): Should return internal
    tracking mechanism used by a tracker class (such as the `run` for wandb)

    Implementations can also include a `main_process_only` (`bool`) attribute to toggle if relevent logging, init, and
    other functions should occur on the main process or across all processes (by default will use `True`)
    """

    main_process_only = True

    def __init__(self, _blank=False):
        if not _blank:
            err = ""
            if not hasattr(self, "name"):
                err += "`name`"
            if not hasattr(self, "requires_logging_directory"):
                if len(err) > 0:
                    err += ", "
                err += "`requires_logging_directory`"

            # as tracker is a @property that relies on post-init
            if "tracker" not in dir(self):
                if len(err) > 0:
                    err += ", "
                err += "`tracker`"
            if len(err) > 0:
                raise NotImplementedError(
                    f"The implementation for this tracker class is missing the following "
                    f"required attributes. Please define them in the class definition: "
                    f"{err}"
                )

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

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = tensorboard.SummaryWriter(self.logging_dir, **kwargs)
        logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        project_run_name = time.time()
        dir_name = os.path.join(self.logging_dir, str(project_run_name))
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(values, outfile)
            except yaml.representer.RepresenterError:
                logger.error("Serialization to store hyperparameters failed")
                raise
        logger.debug("Stored initial configuration hyperparameters to TensorBoard and hparams yaml file")

    @on_main_process
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
        logger.debug("Successfully logged to TensorBoard")

    @on_main_process
    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.debug("TensorBoard writer closed")


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
    main_process_only = False

    @on_main_process
    def __init__(self, run_name: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        self.run = wandb.init(project=self.run_name, **kwargs)
        logger.debug(f"Initialized WandB project {self.run_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        wandb.config.update(values)
        logger.debug("Stored initial configuration hyperparameters to WandB")

    @on_main_process
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
        logger.debug("Successfully logged to WandB")

    @on_main_process
    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.debug("WandB run closed")


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

    @on_main_process
    def __init__(self, run_name: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        self.writer = Experiment(project_name=run_name, **kwargs)
        logger.debug(f"Initialized CometML project {self.run_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.log_parameters(values)
        logger.debug("Stored initial configuration hyperparameters to CometML")

    @on_main_process
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
        logger.debug("Successfully logged to CometML")

    @on_main_process
    def finish(self):
        """
        Closes `comet-ml` writer
        """
        self.writer.end()
        logger.debug("CometML run closed")


class AimTracker(GeneralTracker):
    """
    A `Tracker` class that supports `aim`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `Run.__init__` method.
    """

    name = "aim"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Optional[Union[str, os.PathLike]] = ".", **kwargs):
        self.run_name = run_name
        self.writer = Run(repo=logging_dir, **kwargs)
        self.writer.name = self.run_name
        logger.debug(f"Initialized Aim project {self.run_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        self.writer["hparams"] = values

    @on_main_process
    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `Run.track` method.
        """
        # Note: replace this with the dictionary support when merged
        for key, value in values.items():
            self.writer.track(value, name=key, step=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Closes `aim` writer
        """
        self.writer.close()


class MLflowTracker(GeneralTracker):
    """
    A `Tracker` class that supports `mlflow`. Should be initialized at the start of your script.

    Args:
        experiment_name (`str`, *optional*):
            Name of the experiment. Environment variable MLFLOW_EXPERIMENT_NAME has priority over this argument.
        logging_dir (`str` or `os.PathLike`, defaults to `"."`):
            Location for mlflow logs to be stored.
        run_id (`str`, *optional*):
            If specified, get the run with the specified UUID and log parameters and metrics under that run. The run’s
            end time is unset and its status is set to running, but the run’s other attributes (source_version,
            source_type, etc.) are not changed. Environment variable MLFLOW_RUN_ID has priority over this argument.
        tags (`Dict[str, str]`, *optional*):
            An optional `dict` of `str` keys and values, or a `str` dump from a `dict`, to set as tags on the run. If a
            run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are
            set on the new run. Environment variable MLFLOW_TAGS has priority over this argument.
        nested_run (`bool`, *optional*, defaults to `False`):
            Controls whether run is nested in parent run. True creates a nested run. Environment variable
            MLFLOW_NESTED_RUN has priority over this argument.
        run_name (`str`, *optional*):
            Name of new run (stored as a mlflow.runName tag). Used only when `run_id` is unspecified.
        description (`str`, *optional*):
            An optional string that populates the description box of the run. If a run is being resumed, the
            description is set on the resumed run. If a new run is being created, the description is set on the new
            run.
    """

    name = "mlflow"
    requires_logging_directory = True

    @on_main_process
    def __init__(
        self,
        experiment_name: str = None,
        logging_dir: Optional[Union[str, os.PathLike]] = ".",
        run_id: Optional[str] = None,
        tags: Optional[Union[Dict[str, Any], str]] = None,
        nested_run: Optional[bool] = False,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
        run_id = os.getenv("MLFLOW_RUN_ID", run_id)
        tags = os.getenv("MLFLOW_TAGS", tags)
        if isinstance(tags, str):
            tags = json.loads(tags)

        nested_run = os.getenv("MLFLOW_NESTED_RUN", nested_run)

        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=logging_dir,
            tags=tags,
        )

        self.active_run = mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested_run,
            tags=tags,
            description=description,
        )

        logger.debug(f"Initialized mlflow experiment {experiment_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.active_run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """

        for name, value in list(values.items()):
            # internally, all values are converted to str in MLflow
            if len(str(value)) > mlflow.utils.validation.MAX_PARAM_VAL_LENGTH:
                logger.warning(
                    f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                    f" log_param() only accepts values no longer than {mlflow.utils.validation.MAX_PARAM_VAL_LENGTH} characters so we dropped this attribute."
                )
                del values[name]

        values_list = list(values.items())

        # MLflow cannot log more than 100 values in one go, so we have to split it
        for i in range(0, len(values_list), mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH):
            mlflow.log_params(dict(values_list[i : i + mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH]))

        logger.debug("Stored initial configuration hyperparameters to MLflow")

    @on_main_process
    def log(self, values: dict, step: Optional[int]):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        metrics = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            else:
                logger.warning(
                    f'MLflowTracker is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                    "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                )

        mlflow.log_metrics(metrics, step=step)
        logger.debug("Successfully logged to mlflow")

    @on_main_process
    def finish(self):
        """
        End the active MLflow run.
        """
        mlflow.end_run()


LOGGER_TYPE_TO_CLASS = {
    "aim": AimTracker,
    "comet_ml": CometMLTracker,
    "mlflow": MLflowTracker,
    "tensorboard": TensorBoardTracker,
    "wandb": WandBTracker,
}


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
            - `"mlflow"`
            If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
            also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        logging_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing logs of locally-compatible loggers.
    """
    loggers = []
    if log_with is not None:
        if not isinstance(log_with, (list, tuple)):
            log_with = [log_with]
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
                                        f"Logging with `{log_type}` requires a `logging_dir` to be passed in."
                                    )
                            loggers.append(log_type)
                        else:
                            logger.debug(f"Tried adding logger {log_type}, but package is unavailable in the system.")

    return loggers
