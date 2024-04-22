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

from manim import *


class Stage1(Scene):
    def construct(self):
        mem = Rectangle(height=0.5,width=0.5)
        fill = Rectangle(height=0.46,width=0.46).set_stroke(width=0)

        cpu_left_col_base = [mem.copy() for i in range(6)]
        cpu_right_col_base = [mem.copy() for i in range(6)]
        cpu_left_col = VGroup(*cpu_left_col_base).arrange(UP, buff=0)
        cpu_right_col = VGroup(*cpu_right_col_base).arrange(UP, buff=0)
        cpu_rects = VGroup(cpu_left_col,cpu_right_col).arrange(RIGHT, buff=0)
        cpu_text = Text("CPU", font_size=24)
        cpu = Group(cpu_rects,cpu_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        cpu.move_to([-2.5,-.5,0])
        self.add(cpu)

        gpu_base = [mem.copy() for i in range(1)]
        gpu_rect = VGroup(*gpu_base).arrange(UP,buff=0)
        gpu_text = Text("GPU", font_size=24)
        gpu = Group(gpu_rect,gpu_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        gpu.align_to(cpu, DOWN)
        gpu.set_x(gpu.get_x() - 1)
        
        self.add(gpu)

        model_base = [mem.copy() for i in range(6)]
        model_rect = VGroup(*model_base).arrange(RIGHT,buff=0)

        model_text = Text("Model", font_size=24)
        model = Group(model_rect,model_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        model.move_to([3, -1., 0])
        
        self.play(
            Create(cpu_left_col, run_time=1),
            Create(cpu_right_col, run_time=1),
            Create(gpu_rect, run_time=1),
        )

        step_1 = MarkupText(
            f"First, an empty model skeleton is loaded\ninto <span fgcolor='{YELLOW}'>memory</span> without using much RAM.", 
            font_size=24
        )

        key = Square(side_length=2.2)
        key.move_to([-5, 2, 0])

        key_text = MarkupText(
            f"<b>Key:</b>\n\n<span fgcolor='{YELLOW}'>●</span> Empty Model",
            font_size=18,
        )

        key_text.move_to([-5, 2.4, 0])


        step_1.move_to([2, 2, 0])
        self.play(
            Write(step_1, run_time=2.5),
            Write(key_text),
            Write(key)
        )

        self.add(model)
        

        cpu_targs = []
        first_animations = []
        second_animations = []
        for i,rect in enumerate(model_base):

            cpu_target = Rectangle(height=0.46,width=0.46).set_stroke(width=0.).set_fill(YELLOW, opacity=0.7)
            cpu_target.move_to(rect)
            cpu_target.generate_target()
            cpu_target.target.height = 0.46/4
            cpu_target.target.width = 0.46/3
            
            if i == 0:
                cpu_target.target.next_to(cpu_left_col_base[0].get_corner(DOWN+LEFT), buff=0.02, direction=UP)
                cpu_target.target.set_x(cpu_target.target.get_x()+0.1)
            elif i == 3:
                cpu_target.target.next_to(cpu_targs[0].target, direction=UP, buff=0.)
            else:
                cpu_target.target.next_to(cpu_targs[i-1].target, direction=RIGHT, buff=0.)
            cpu_targs.append(cpu_target)

            first_animations.append(rect.animate(run_time=0.5).set_stroke(YELLOW))
            second_animations.append(MoveToTarget(cpu_target, run_time=1.5))

        self.play(*first_animations)
        self.play(*second_animations)
                 

        self.wait()