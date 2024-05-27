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


class Stage6(Scene):
    def construct(self):
        # The dataset items
        colors = ["BLUE_E", "DARK_BROWN", "GOLD_E", "GRAY_A"]
        fill = Rectangle(height=0.46,width=0.46).set_stroke(width=0)
        columns = [
            VGroup(*[Rectangle(height=0.25,width=0.25,color=colors[j]) for i in range(8)]).arrange(RIGHT,buff=0)
            for j in range(4)
        ]
        dataset_recs = VGroup(*columns).arrange(UP, buff=0)
        dataset_text = Text("Dataset", font_size=24)
        dataset = Group(dataset_recs,dataset_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        dataset.move_to([-2,0,0])
        self.add(dataset)
        code = Code(
            code="# We enable this by default\naccelerator = Accelerator()\ndataloader = DataLoader(..., shuffle=True)\ndataloader = accelerator.prepare(dataloader)\nfor batch in dataloader:\n\t...",
            tab_width=4,
            background="window",
            language="Python",
            font="Monospace",
            font_size=14,
            corner_radius=.2,
            insert_line_no=False,
            line_spacing=.75,
            style=Code.styles_list[1],
        )
        code.move_to([-3.5, 2.5, 0])
        self.add(code)

        # The dataloader itself

        sampler_1 = Group(
            Rectangle(color="blue", height=1, width=1),
            Text("Sampler GPU 1", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN)
        sampler_2 = Group(
            Rectangle(color="blue", height=1, width=1),
            Text("Sampler GPU 2", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN)
        sampler_3 = Group(
            Rectangle(color="blue", height=1, width=1),
            Text("Sampler GPU 3", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN)
        sampler_4 = Group(
            Rectangle(color="blue", height=1, width=1),
            Text("Sampler GPU 4", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN)
        sampler_1.move_to([2,2,0])
        sampler_2.move_to([2,.5,0])
        sampler_3.move_to([2,-1.,0])
        sampler_4.move_to([2,-2.5,0])
        self.add(sampler_1, sampler_2, sampler_3, sampler_4)
        samplers = [sampler_1[0], sampler_2[0], sampler_3[0], sampler_4[0]]

        gpu_1 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("Output GPU 1", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4.5, 2, 0])
        gpu_2 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("Output GPU 2", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4.5, .5, 0])
        gpu_3 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("Output GPU 3", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4.5, -1, 0])
        gpu_4 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("Output GPU 4", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4.5, -2.5, 0])
        gpus = [gpu_1[0], gpu_2[0], gpu_3[0], gpu_4[0]]
        self.add(gpu_1, gpu_2, gpu_3, gpu_4)


        first_animations = []
        second_animations = []


        colors = ["BLUE_E", "DARK_BROWN", "GOLD_E", "GRAY_A"]
        current_color = colors[0]
        buff = 0
        lr_buff = .25
        old_target = None
        new_datasets = []
        for i,row_data in enumerate(dataset_recs):
            new_row = []
            current_color = colors[i]
            if i == 0:
                idx = -3
            elif i == 1:
                idx = -2
            elif i == 2:
                idx = -1
            elif i == 3:
                idx = 0
            for j,indiv_data in enumerate(row_data):
                dataset_target = Rectangle(height=0.46/2,width=0.46/2).set_stroke(width=0.).set_fill(current_color, opacity=0.7)
                dataset_target.move_to(indiv_data)
                dataset_target.generate_target()
                aligned_edge = ORIGIN
                if j % 8 == 0:
                    aligned_edge = LEFT
                    old_target = dataset_target.target
                    dataset_target.target.next_to(
                        samplers[abs(idx)].get_corner(UP+LEFT), buff=.02, direction=RIGHT+DOWN,
                    )
                    dataset_target.target.set_x(dataset_target.target.get_x())
                elif j % 4 == 0:
                    old_target = dataset_target.target
                    dataset_target.target.next_to(
                        samplers[abs(idx)].get_corner(UP+LEFT), buff=.02, direction=RIGHT+DOWN,
                    )
                    dataset_target.target.set_x(dataset_target.target.get_x())
                    dataset_target.target.set_y(dataset_target.target.get_y()-.25)
                else:
                    dataset_target.target.next_to(
                        old_target, direction=RIGHT, buff=0.02,
                    )
                old_target = dataset_target.target
                new_row.append(dataset_target)
                first_animations.append(indiv_data.animate(run_time=0.5).set_stroke(current_color))
                second_animations.append(MoveToTarget(dataset_target, run_time=1.5))
            
            new_datasets.append(new_row)
        step_1 = MarkupText(
            f"During shuffling, each mini-batch's\noutput order will be modified",
            font_size=18
        )
        step_1.move_to([-1.5, -2, 0])

        self.play(
            Write(step_1, run_time=3),
        )
        self.play(
            *first_animations,
        )
        self.play(*second_animations)
        self.wait(duration=.5)

        move_animation = []
        import random
        for i,row in enumerate(new_datasets):
            row = [row[k] for k in random.sample(range(8), 8)]
            current_color = colors[i]
            if i == 0:
                idx = -3
            elif i == 1:
                idx = -2
            elif i == 2:
                idx = -1
            elif i == 3:
                idx = 0
            for j,indiv_data in enumerate(row):
                indiv_data.generate_target()
                aligned_edge = ORIGIN
                if j % 8 == 0:
                    aligned_edge = LEFT
                    indiv_data.target.next_to(
                        gpus[abs(idx)].get_corner(UP+LEFT), buff=.02, direction=RIGHT+DOWN,
                    )
                    indiv_data.target.set_x(indiv_data.target.get_x())
                elif j % 4 == 0:
                    indiv_data.target.next_to(
                        gpus[abs(idx)].get_corner(UP+LEFT), buff=.02, direction=RIGHT+DOWN,
                    )
                    indiv_data.target.set_x(indiv_data.target.get_x())
                    indiv_data.target.set_y(indiv_data.target.get_y()-.25)
                else:
                    indiv_data.target.next_to(
                        old_target, direction=RIGHT, buff=0.02,
                    )
                old_target = indiv_data.target
                move_animation.append(MoveToTarget(indiv_data, run_time=1.5))

        self.play(*move_animation)
        self.wait()