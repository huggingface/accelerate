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

class Stage2(Scene):
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

        gpu_base = [mem.copy() for i in range(4)]
        gpu_rect = VGroup(*gpu_base).arrange(UP,buff=0)
        gpu_text = Text("GPU", font_size=24)
        gpu = Group(gpu_rect,gpu_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        gpu.move_to([-1,-1,0])
        self.add(gpu)

        model_base = [mem.copy() for i in range(6)]
        model_rect = VGroup(*model_base).arrange(RIGHT,buff=0)

        model_text = Text("Model", font_size=24)
        model = Group(model_rect,model_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        model.move_to([3, -1., 0])
        self.add(model)
        
        cpu_targs = []
        for i,rect in enumerate(model_base):
            rect.set_stroke(YELLOW)
            # target = fill.copy().set_fill(YELLOW, opacity=0.7)
            # target.move_to(rect)
            # self.add(target)

            cpu_target = Rectangle(height=0.46/4,width=0.46/3).set_stroke(width=0.).set_fill(YELLOW, opacity=0.7)
            
            if i == 0:
                cpu_target.next_to(cpu_left_col_base[0].get_corner(DOWN+LEFT), buff=0.02, direction=UP)
                cpu_target.set_x(cpu_target.get_x()+0.1)
            elif i == 3:
                cpu_target.next_to(cpu_targs[0], direction=UP, buff=0.)
            else:
                cpu_target.next_to(cpu_targs[i-1], direction=RIGHT, buff=0.)
            self.add(cpu_target)
            cpu_targs.append(cpu_target)

              

        checkpoint_base = [mem.copy() for i in range(6)]
        checkpoint_rect = VGroup(*checkpoint_base).arrange(RIGHT,buff=0)

        checkpoint_text = Text("Loaded Checkpoint", font_size=24)
        checkpoint = Group(checkpoint_rect,checkpoint_text).arrange(DOWN, aligned_edge=DOWN, buff=0.4)
        checkpoint.move_to([3, .5, 0])
            
        key = Square(side_length=2.2)
        key.move_to([-5, 2, 0])

        key_text = MarkupText(
            f"<b>Key:</b>\n\n<span fgcolor='{YELLOW}'>●</span> Empty Model",
            font_size=18,
        )

        key_text.move_to([-5, 2.4, 0])

        self.add(key_text, key)

        blue_text = MarkupText(
            f"<span fgcolor='{BLUE}'>●</span> Checkpoint",
            font_size=18,
        )

        blue_text.next_to(key_text, DOWN*2.4, aligned_edge=key_text.get_left())

        step_2 = MarkupText(
            f'Next, a <i><span fgcolor="{BLUE}">second</span></i> model is loaded into memory,\nwith the weights of a <span fgcolor="{BLUE}">single shard</span>.', 
            font_size=24
        )
        step_2.move_to([2, 2, 0])
        self.play(
            Write(step_2),
            Write(blue_text)
        )

        self.play(
            Write(checkpoint_text, run_time=1),
            Create(checkpoint_rect, run_time=1)
        )

        first_animations = []
        second_animations = []
        for i,rect in enumerate(checkpoint_base):
            target = fill.copy().set_fill(BLUE, opacity=0.7)
            target.move_to(rect)
            first_animations.append(GrowFromCenter(target, run_time=1))

            cpu_target = target.copy()
            cpu_target.generate_target()
            if i < 5:
                cpu_target.target.move_to(cpu_left_col_base[i+1])
            else:
                cpu_target.target.move_to(cpu_right_col_base[i-5])
            second_animations.append(MoveToTarget(cpu_target, run_time=1.5))
            
        self.play(*first_animations)
        self.play(*second_animations)
        self.wait()