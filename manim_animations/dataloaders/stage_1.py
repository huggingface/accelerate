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

%%manim -qh -v WARNING Stage01
class Stage01(Scene):
    def construct(self):
        mascot = ImageMobject("mascot_bookie.png")
        mascot.scale(.35)
        mascot.move_to([-3.75,-1,0])
        text = Paragraph(
            "Distributed Training,\nHugging Face Accelerate,\nand PyTorch DataLoaders\n\nHow do they all interact?", 
            font_size=36,
            line_spacing=1,
            alignment="center",
            weight=BOLD,
        )
        text.move_to([1.75,.5,0])
        self.add(mascot)
        self.add(text)