# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import argparse


class _StoreAction(argparse.Action):
    """
    Custom action that allows for `-` or `_` to be passed in for an argument.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new_option_strings = []
        for option_string in self.option_strings:
            new_option_strings.append(option_string)
            if "_" in option_string[2:]:
                # Add `-` version to the option string
                new_option_strings.append(option_string.replace("_", "-"))
        self.option_strings = new_option_strings

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _StoreTrueAction(_StoreAction):
    """
    Same as `argparse._StoreTrueAction` but uses the custom `_StoreAction`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, const=True)


class CustomArgumentParser(argparse.ArgumentParser):
    """
    Custom argument parser that allows for the use of `-` or `_` in arguments passed and overrides the help for each
    when applicable.
    """

    def add_argument(self, *args, **kwargs):
        if "action" in kwargs:
            # Translate action -> class
            if kwargs["action"] == "store_true":
                kwargs["action"] = _StoreTrueAction
        else:
            kwargs["action"] = _StoreAction
        super().add_argument(*args, **kwargs)
