#!/usr/bin/env python

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
"""
A collection of utilities for comparing `examples/complete_*_example.py` scripts with the capabilities inside of each
`examples/by_feature` example. `compare_against_test` is the main function that should be used when testing, while the
others are used to either get the code that matters, or to preprocess them (such as stripping comments)
"""

import os


def get_function_contents_by_name(lines: list[str], name: str):
    """
    Extracts a function from `lines` of segmented source code with the name `name`.

    Args:
        lines (`List[str]`):
            Source code of a script separated by line.
        name (`str`):
            The name of the function to extract. Should be either `training_function` or `main`
    """
    if name != "training_function" and name != "main":
        raise ValueError(f"Incorrect function name passed: {name}, choose either 'main' or 'training_function'")
    good_lines, found_start = [], False
    for line in lines:
        if not found_start and f"def {name}" in line:
            found_start = True
            good_lines.append(line)
            continue
        if found_start:
            if name == "training_function" and "def main" in line:
                return good_lines
            if name == "main" and "if __name__" in line:
                return good_lines
            good_lines.append(line)


def clean_lines(lines: list[str]):
    """
    Filters `lines` and removes any entries that start with a comment ('#') or is just a newline ('\n')

    Args:
        lines (`List[str]`):
            Source code of a script separated by line.
    """
    return [line for line in lines if not line.lstrip().startswith("#") and line != "\n"]


def compare_against_test(base_filename: str, feature_filename: str, parser_only: bool, secondary_filename: str = None):
    """
    Tests whether the additional code inside of `feature_filename` was implemented in `base_filename`. This should be
    used when testing to see if `complete_*_.py` examples have all of the implementations from each of the
    `examples/by_feature/*` scripts.

    It utilizes `nlp_example.py` to extract out all of the repeated training code, so that only the new additional code
    is examined and checked. If something *other* than `nlp_example.py` should be used, such as `cv_example.py` for the
    `complete_cv_example.py` script, it should be passed in for the `secondary_filename` parameter.

    Args:
        base_filename (`str` or `os.PathLike`):
            The filepath of a single "complete" example script to test, such as `examples/complete_cv_example.py`
        feature_filename (`str` or `os.PathLike`):
            The filepath of a single feature example script. The contents of this script are checked to see if they
            exist in `base_filename`
        parser_only (`bool`):
            Whether to compare only the `main()` sections in both files, or to compare the contents of
            `training_loop()`
        secondary_filename (`str`, *optional*):
            A potential secondary filepath that should be included in the check. This function extracts the base
            functionalities off of "examples/nlp_example.py", so if `base_filename` is a script other than
            `complete_nlp_example.py`, the template script should be included here. Such as `examples/cv_example.py`
    """
    with open(base_filename) as f:
        base_file_contents = f.readlines()
    with open(os.path.abspath(os.path.join("examples", "nlp_example.py"))) as f:
        full_file_contents = f.readlines()
    with open(feature_filename) as f:
        feature_file_contents = f.readlines()
    if secondary_filename is not None:
        with open(secondary_filename) as f:
            secondary_file_contents = f.readlines()

    # This is our base, we remove all the code from here in our `full_filename` and `feature_filename` to find the new content
    if parser_only:
        base_file_func = clean_lines(get_function_contents_by_name(base_file_contents, "main"))
        full_file_func = clean_lines(get_function_contents_by_name(full_file_contents, "main"))
        feature_file_func = clean_lines(get_function_contents_by_name(feature_file_contents, "main"))
        if secondary_filename is not None:
            secondary_file_func = clean_lines(get_function_contents_by_name(secondary_file_contents, "main"))
    else:
        base_file_func = clean_lines(get_function_contents_by_name(base_file_contents, "training_function"))
        full_file_func = clean_lines(get_function_contents_by_name(full_file_contents, "training_function"))
        feature_file_func = clean_lines(get_function_contents_by_name(feature_file_contents, "training_function"))
        if secondary_filename is not None:
            secondary_file_func = clean_lines(
                get_function_contents_by_name(secondary_file_contents, "training_function")
            )

    _dl_line = "train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)\n"

    # Specific code in our script that differs from the full version, aka what is new
    new_feature_code = []
    passed_idxs = []  # We keep track of the idxs just in case it's a repeated statement
    it = iter(feature_file_func)
    for i in range(len(feature_file_func) - 1):
        if i not in passed_idxs:
            line = next(it)
            if (line not in full_file_func) and (line.lstrip() != _dl_line):
                if "TESTING_MOCKED_DATALOADERS" not in line:
                    new_feature_code.append(line)
                    passed_idxs.append(i)
                else:
                    # Skip over the `config['num_epochs'] = 2` statement
                    _ = next(it)

    # Extract out just the new parts from the full_file_training_func
    new_full_example_parts = []
    passed_idxs = []  # We keep track of the idxs just in case it's a repeated statement
    for i, line in enumerate(base_file_func):
        if i not in passed_idxs:
            if (line not in full_file_func) and (line.lstrip() != _dl_line):
                if "TESTING_MOCKED_DATALOADERS" not in line:
                    new_full_example_parts.append(line)
                    passed_idxs.append(i)

    # Finally, get the overall diff
    diff_from_example = [line for line in new_feature_code if line not in new_full_example_parts]
    if secondary_filename is not None:
        diff_from_two = [line for line in full_file_contents if line not in secondary_file_func]
        diff_from_example = [line for line in diff_from_example if line not in diff_from_two]

    return diff_from_example
