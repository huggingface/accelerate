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

from manim import *

class Stage3(Scene):
    def construct(self):
        step_1 = MarkupText(
            f"To combat this, Accelerate employs one of two different\nSampler wrapper methods depending on the scenario:",
            font_size=24
        )
        step_1.move_to([0, 1.5, 0])
        self.add(step_1)
        step_2 = MarkupText(
            f"1. Sharding the dataset before drawing:\n\t● <span fgcolor='{RED}'>IterableDatasetShard</span>\n\t● <span fgcolor='{RED}'>BatchSamplerShard</span>",
            font_size=24,
        ).next_to(step_1, direction=DOWN, aligned_edge=LEFT)
        self.add(step_2)
        step_3 = MarkupText(
            f"\n\n2. Splitting the batch after drawing:\n\t● <span fgcolor='{BLUE}'>DataLoaderDispatcher</span>",
            font_size=24,
        ).next_to(step_2, direction=DOWN, aligned_edge=LEFT)
        self.add(step_3)