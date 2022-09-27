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

class Stage5(Scene):
    def construct(self):
        mem = Rectangle(height=0.5,width=0.5)
        fill = Rectangle(height=0.46,width=0.46).set_stroke(width=0)

        meta_mem = Rectangle(height=0.25,width=0.25)

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

        model_arr = []
        model_cpu_arr = []
        
        for i,rect in enumerate(model_base):
            target = fill.copy().set_fill(BLUE, opacity=0.8)
            target.move_to(rect)
            model_arr.append(target)

            cpu_target = Rectangle(height=0.46,width=0.46).set_stroke(width=0.).set_fill(BLUE, opacity=0.8)
            cpu_target.move_to(cpu_left_col_base[i])
            model_cpu_arr.append(cpu_target)

        self.add(*model_arr, *model_cpu_arr)

        disk_left_col_base = [meta_mem.copy() for i in range(6)]
        disk_right_col_base = [meta_mem.copy() for i in range(6)]
        disk_left_col = VGroup(*disk_left_col_base).arrange(UP, buff=0)
        disk_right_col = VGroup(*disk_right_col_base).arrange(UP, buff=0)
        disk_rects = VGroup(disk_left_col,disk_right_col).arrange(RIGHT, buff=0)
        disk_text = Text("Disk", font_size=24)
        disk = Group(disk_rects,disk_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        disk.move_to([-4,-1.25,0])
        self.add(disk_text, disk_rects)

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
        self.add(blue_text)

        step_6 = MarkupText(
            f'Now watch as an input is passed through the model\nand how the memory is utilized and handled.', 
            font_size=24
        )
        step_6.move_to([2, 2, 0])

        self.play(Write(step_6))

        input = Square(0.3)
        input.set_fill(RED, opacity=1.)
        input.set_stroke(width=0.)
        input.next_to(model_base[0], LEFT, buff=.5)

        self.play(Write(input))

        input.generate_target()
        input.target.next_to(model_arr[0], direction=LEFT, buff=0.02)
        self.play(MoveToTarget(input))

        self.play(FadeOut(step_6))


        a = Arrow(start=UP, end=DOWN, color=RED, buff=.5)
        a.next_to(model_arr[0].get_left(), UP, buff=0.2)

        model_cpu_arr[0].generate_target()
        model_cpu_arr[0].target.move_to(gpu_rect[0])

        step_7 = MarkupText(
            f'As the input reaches a layer, the hook triggers\nand weights are moved from the CPU\nto the GPU and back.', 
            font_size=24
        )
        step_7.move_to([2, 2, 0])

        self.play(Write(step_7, run_time=3))

        circ_kwargs = {"run_time":1, "fade_in":True, "fade_out":True, "buff":0.02}

        self.play(
            Write(a), 
            Circumscribe(model_arr[0], color=ORANGE, **circ_kwargs),
            Circumscribe(model_cpu_arr[0], color=ORANGE, **circ_kwargs),
            Circumscribe(gpu_rect[0], color=ORANGE, **circ_kwargs),
        )
        self.play(
            MoveToTarget(model_cpu_arr[0])
        )

        a_c = a.copy()
        for i in range(6):
            a_c.next_to(model_arr[i].get_right()+0.02, UP, buff=0.2)

            input.generate_target()
            input.target.move_to(model_arr[i].get_right()+0.02)

            grp = AnimationGroup(
                FadeOut(a, run_time=.5), 
                MoveToTarget(input, run_time=.5), 
                FadeIn(a_c, run_time=.5),
                lag_ratio=0.2
            )

            self.play(grp)


            model_cpu_arr[i].generate_target()
            model_cpu_arr[i].target.move_to(cpu_left_col_base[i])


            if i < 5:
                model_cpu_arr[i+1].generate_target()
                model_cpu_arr[i+1].target.move_to(gpu_rect[0])
                if i >= 1:
                    circ_kwargs["run_time"] = .7

                self.play(
                    Circumscribe(model_arr[i], **circ_kwargs),
                    Circumscribe(cpu_left_col_base[i], **circ_kwargs),
                    Circumscribe(cpu_left_col_base[i+1], color=ORANGE, **circ_kwargs),                    
                    Circumscribe(gpu_rect[0], color=ORANGE, **circ_kwargs),
                    Circumscribe(model_arr[i+1], color=ORANGE, **circ_kwargs),
                )
                if i < 1:
                    self.play(
                        MoveToTarget(model_cpu_arr[i]), 
                        MoveToTarget(model_cpu_arr[i+1]),
                    )
                else:
                    self.play(
                        MoveToTarget(model_cpu_arr[i], run_time=.7), 
                        MoveToTarget(model_cpu_arr[i+1], run_time=.7),
                    )
            else:
                model_cpu_arr[i].generate_target()
                model_cpu_arr[i].target.move_to(cpu_left_col_base[-1])
                input.generate_target()
                input.target.next_to(model_arr[-1].get_right(), RIGHT+0.02, buff=0.2)

                self.play(
                    Circumscribe(model_arr[-1], color=ORANGE, **circ_kwargs),
                    Circumscribe(cpu_left_col_base[-1], color=ORANGE, **circ_kwargs),
                    Circumscribe(gpu_rect[0], color=ORANGE, **circ_kwargs),
                )

                self.play(
                    MoveToTarget(model_cpu_arr[i])
                )

            a = a_c
            a_c = a_c.copy()

        input.generate_target()
        input.target.next_to(model_base[-1], RIGHT+0.02, buff=.5)
        self.play(
            FadeOut(step_7),
            FadeOut(a, run_time=.5), 
        )

        step_8 = MarkupText(
            f'Inference on a model too large for GPU memory\nis successfully completed.', font_size=24
        )
        step_8.move_to([2, 2, 0])

        self.play(
            Write(step_8, run_time=3),
            MoveToTarget(input)
        )

        self.wait()