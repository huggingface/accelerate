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


class Stage2(Scene):
    def construct(self):
        # The dataset items
        fill = Rectangle(height=0.46,width=0.46).set_stroke(width=0)
        columns = [
            VGroup(*[Rectangle(height=0.25,width=0.25,color="green") for i in range(8)]).arrange(RIGHT,buff=0)
            for j in range(4)
        ]
        dataset_recs = VGroup(*columns).arrange(UP, buff=0)
        dataset_text = Text("Dataset", font_size=24)
        dataset = Group(dataset_recs,dataset_text).arrange(DOWN, buff=0.5, aligned_edge=DOWN)
        dataset.move_to([-2,0,0])
        self.add(dataset)
        
        code = Code(
            code="dataloader = DataLoader(...)\nfor batch in dataloader():\n\t...",
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
        dataloader = Group(
            Rectangle(color="red", height=2, width=2),
            Text("DataLoader", font_size=24)
        ).arrange(DOWN, buff=.5, aligned_edge=DOWN)

        sampler = Group(
            Rectangle(color="blue", height=1, width=1),
            Text("Sampler", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN)
        dataloader.move_to([1, 0, 0])
        sampler.move_to([.75,.25,0])
        self.add(dataloader)
        self.add(sampler)

        gpu_1 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("GPU 1", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4, 2, 0])
        gpu_2 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("GPU 2", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4, .5, 0])
        gpu_3 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("GPU 3", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4, -1, 0])
        gpu_4 = Group(
            Rectangle(color="white", height=1, width=1),
            Text("GPU 4", font_size=12)
        ).arrange(DOWN, buff=.25, aligned_edge=DOWN).move_to([4, -2.5, 0])
        gpus = [gpu_1[0], gpu_2[0], gpu_3[0], gpu_4[0]]
        self.add(gpu_1, gpu_2, gpu_3, gpu_4)

        # Animate their existence
        self.play(
            Create(gpu_1[0], run_time=0.5),
            Create(gpu_2[0], run_time=0.5),
            Create(gpu_3[0], run_time=0.5),
            Create(gpu_4[0], run_time=0.5),
            Create(dataset_recs, run_time=1),
            Create(sampler[0], run_time=1),
            Create(dataloader[0], run_time=1)
        )

        step_1 = MarkupText(
            f"Without any special care, \nthe same data is sent though each sampler, \nand the same samples are spit out on each GPU",
            font_size=18
        )
        step_1.move_to([0, -2.5, 0])
        self.play(
            Write(step_1, run_time=4),
        )

        first_animations = []
        second_animations = []


        colors = ["BLUE_E", "DARK_BROWN", "GOLD_E", "GRAY_A"]
        current_color = colors[0]
        buff = 0
        lr_buff = .25
        old_target = None
        new_datasets = []
        for i,data in enumerate(dataset_recs[-1]):
            if i % 2 == 0:
                # current_color = colors[i//2]
                current_color = "BLUE_E"
            dataset_target = Rectangle(height=0.46/2,width=0.46/2).set_stroke(width=0.).set_fill(current_color, opacity=0.7)
            dataset_target.move_to(data)
            dataset_target.generate_target()
            aligned_edge = ORIGIN
            if i % 2 == 0:
                old_target = dataset_target.target
                buff -= .25
                aligned_edge = LEFT
                dataset_target.target.next_to(
                    sampler, buff=buff, direction=UP,
                    aligned_edge=LEFT
                )
            else:
                dataset_target.target.next_to(
                    old_target, direction=RIGHT, buff=0.01,
                )
            new_datasets.append(dataset_target)
            first_animations.append(data.animate(run_time=0.5).set_stroke(current_color))
            second_animations.append(MoveToTarget(dataset_target, run_time=1.5))
        self.play(*first_animations)
        self.play(*second_animations)
        self.wait()

        move_animation = []

        for j,gpu in enumerate(gpus):
            buff = 0
            for i,data in enumerate(new_datasets):
                if i % 2 == 0:
                    current_color = colors[i//2]
                if j != 3:
                    data = data.copy()
                data.generate_target()
                aligned_edge = ORIGIN
                if i % 2 == 0:
                    old_target = data.target
                    buff -= .25
                    aligned_edge = LEFT
                    data.target.next_to(
                        gpu, buff=buff, direction=UP,
                        aligned_edge=LEFT
                    )
                else:
                    data.target.next_to(
                        old_target, direction=RIGHT, buff=0.01,
                    )
                move_animation.append(MoveToTarget(data, run_time=1.5))


        self.play(*move_animation)

        self.remove(step_1)
        step_2 = MarkupText(
            f"This behavior is undesireable, because we want\neach GPU to see different data for efficient training.",
            font_size=18
        )
        step_2.move_to([0, -2.5, 0])

        self.play(
            Write(step_2, run_time=2.5),
        )
        self.wait()