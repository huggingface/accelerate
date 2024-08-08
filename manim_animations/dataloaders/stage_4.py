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

class Stage4(Scene):
    def construct(self):

        step_1 = MarkupText(
            f"To understand the next part fully, let's define two terms,\n<span fgcolor='{RED}'>`batch_size`</span> and <span fgcolor='{BLUE}'>`global_batch_size`</span>:",
            font_size=18
        )
        step_1.move_to([0, 1.5, 0])
        # <span fgcolor='{YELLOW}'>●</span>
        step_2 = MarkupText(
            f"\n\n● <span fgcolor='{RED}'>`batch_size`</span>: \n\tThis will be defined as the batch size seen on a given\n\t*individual* GPU",
            font_size=18,
        ).next_to(step_1, direction=DOWN, aligned_edge=LEFT)

        step_3 = MarkupText(
            f"\n\n● <span fgcolor='{BLUE}'>`global_batch_size`</span>:\n\tThis will be defined as the *total* number of\n\tdifferent items seen in the dataset, across all GPUs",
            font_size=18,
        ).next_to(step_2, direction=DOWN, aligned_edge=LEFT)

        step_4 = MarkupText(
            f"\n\nSo if we have a dataset of 64 items, 8 GPUs, \nand a `batch_size` of 8, each *step* will go through\nthe entire dataset one time as 8*8=64",
            font_size=18,
        ).next_to(step_3, direction=DOWN, aligned_edge=LEFT)
        self.play(
            Write(step_1, run_time=4),
        )
        self.play(
            Write(step_2, run_time=4)
        )
        self.play(
            Write(step_3, run_time=4)
        )
        self.play(
            Write(step_4, run_time=6)
        )
        self.wait()