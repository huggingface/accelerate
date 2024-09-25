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

"""
A base script which outputs the accelerate config for the given environment
"""

from accelerate import Accelerator


accelerator = Accelerator()

accelerator.print(f"Accelerator state from the current environment:\n{accelerator.state}")
if accelerator.fp8_recipe_handler is not None:
    accelerator.print(f"FP8 config:\n{accelerator.fp8_recipe_handler}")
accelerator.end_training()
