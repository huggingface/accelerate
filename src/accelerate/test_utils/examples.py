#!/usr/bin/env python

# Copyright 2021 The HuggingFace Team. All rights reserved.
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


logger = logging.getLogger(__name__)


def get_train_func(lines: list):
    """
    Finds the main training function from inside segmented source code.

    Args:
        lines (`list`):
            Source code of a script
    """
    good_lines, found_start = [], False
    for line in lines:
        if not found_start and "def training_function" in line:
            found_start = True
            good_lines.append(line)
            continue
        if found_start:
            if "def main" in line:
                return good_lines
            good_lines.append(line)


def get_main_func(lines: list):
    """
    Finds the main function from inside segmented source code

    Args:
        lines (`list`):
            Source code of a script
    """
    good_lines, found_start = [], False
    for line in lines:
        if not found_start and "def main" in line:
            found_start = True
            good_lines.append(line)
            continue
        if found_start:
            if "if __name__" in line:
                return good_lines
            good_lines.append(line)


def clean_lines(lines: list):
    """
    Filters `lines` and removes any entries that start with a comment ('#') or is just a newline ('\n')

    Args:
        lines (`list`):
            Source code of a script
    """
    return [line for line in lines if not line.lstrip().startswith("#") and line != "\n"]


def compare_against_test(base_filename: str, feature_filename: str, parser_only: bool):
    """
    Checks whether the content aligned in `test_filename` is included inside of `full_filename`.

    If not, will return the different lines between the two.

    Args:
        base_filename (`str`):
            The base template for the script, such as `nlp_example.py` or `cv_example.py`
        feature_filename (`str`):
            The script located in `by_feature` where we want to verify its behavior is mimicked in `full_filename`
        parser_only (`bool`):
            Whether to compare against the contents of `main()` or `training_function()`
    """
    with open(base_filename, "r") as f:
        base_file_contents = f.readlines()
    with open(os.path.abspath(os.path.join("examples", "nlp_example.py")), "r") as f:
        full_file_contents = f.readlines()
    with open(feature_filename, "r") as f:
        feature_file_contents = f.readlines()

    # This is our base, we remove all the code from here in our `full_filename` and `feature_filename` to find the new content
    if parser_only:
        base_file_func = clean_lines(get_main_func(base_file_contents))
        full_file_func = clean_lines(get_main_func(full_file_contents))
        feature_file_func = clean_lines(get_main_func(feature_file_contents))
    else:
        base_file_func = clean_lines(get_train_func(base_file_contents))
        full_file_func = clean_lines(get_train_func(full_file_contents))
        feature_file_func = clean_lines(get_train_func(feature_file_contents))

    _dl_line = "train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)\n"

    # Specific code in our script that differs from the base, aka what is new
    new_feature_code = [
        line for line in feature_file_func if (line not in base_file_func) and (line.lstrip() != _dl_line)
    ]

    # Extract out just the new parts from the full_file_training_func
    new_full_example_parts = [
        line for line in full_file_func if (line not in base_file_func) and (line.lstrip() != _dl_line)
    ]

    # Finally, get the overall diff
    diff_from_example = [line for line in new_feature_code if line not in new_full_example_parts]
    return diff_from_example
