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

from rich import get_console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.traceback import install


console = get_console()


install(show_locals=False)

_kind2prompt = {"bool": Confirm.ask, "float": FloatPrompt.ask, "int": IntPrompt.ask, "default": Prompt.ask}


def _ask_prompt(prompt, kind="default", choices=None, default=None, password=False):
    if default is None:
        if kind == "bool":
            default = False
        elif kind == "int":
            default = "0"
    show_choices = prompt.find("[0]") == -1
    return _kind2prompt[kind](prompt, default=default, choices=choices, show_choices=show_choices, password=password)
