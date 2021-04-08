import inspect
import os
import sys
import unittest
from dataclasses import dataclass

from accelerate.commands.launch import _convert_nargs_to_dict


@dataclass
class MockLaunchConfig:
    training_script_args = [
        "--model_name_or_path",
        "bert",
        "--do_train",
        "--do_test",
        "False",
        "--do_predict",
        "--epochs",
        "3",
        "--learning_rate",
        "5e-5",
        "--max_steps",
        "50.5",
    ]


# class SageMakerConfig(unittest.TestCase):
#     def test_kwargs_handler(self):
#         # If no defaults are changed, `to_kwargs` returns an empty dict.
#         self.assertDictEqual(MockClass().to_kwargs(), {})
#         self.assertDictEqual(MockClass(a=2).to_kwargs(), {"a": 2})
#         self.assertDictEqual(MockClass(a=2, b=True).to_kwargs(), {"a": 2, "b": True})
#         self.assertDictEqual(MockClass(a=2, c=2.25).to_kwargs(), {"a": 2, "c": 2.25})


class SageMakerLaunch(unittest.TestCase):
    def test_args_convert(self):
        # If no defaults are changed, `to_kwargs` returns an empty dict.
        converted_args = _convert_nargs_to_dict(MockLaunchConfig.training_script_args)
        assert isinstance(converted_args["model_name_or_path"], str)
        assert isinstance(converted_args["do_train"], bool)
        assert isinstance(converted_args["do_test"], bool)
        assert isinstance(converted_args["do_predict"], bool)
        assert isinstance(converted_args["epochs"], int)
        assert isinstance(converted_args["learning_rate"], float)
        assert isinstance(converted_args["max_steps"], float)
