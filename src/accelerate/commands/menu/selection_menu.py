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

"""
Main driver for the selection menu, based on https://github.com/bchao1/bullet
"""
from . import cursor, input, keymap
from .helpers import Direction, clear_line, forceWrite, move_cursor, reset_cursor


@input.register
class BulletMenu:
    """
    A CLI menu to select a choice from a list of choices using the keyboard.
    """

    def __init__(self, prompt: str = None, choices: list = []):
        self.position = 0
        self.choices = choices
        self.prompt = prompt

    def print_choice(self, index: int):
        "Prints the choice at the given index"
        if index == self.position:
            forceWrite(f" * {self.choices[index]}")
        else:
            forceWrite(f"   {self.choices[index]}")
        reset_cursor()

    def move_direction(self, direction: Direction):
        "Should not be directly called, used to move a direction of either up or down"
        clear_line()
        old_position = self.position
        if direction == Direction.DOWN:
            if self.position + 1 >= len(self.choices):
                return
            self.position += 1
        else:
            if self.position - 1 < 0:
                return
            self.position -= 1
        self.print_choice(old_position)
        move_cursor(1, direction.name)
        self.print_choice(self.position)

    @input.mark(keymap.ARROW_UP_KEY)
    def move_up(self):
        self.move_direction(Direction.UP)

    @input.mark(keymap.ARROW_DOWN_KEY)
    def move_down(self):
        self.move_direction(Direction.DOWN)

    @input.mark(keymap.NEWLINE_KEY)
    def select(self):
        move_cursor(len(self.choices) - self.position, "DOWN")
        return self.position

    @input.mark(keymap.INTERRUPT_KEY)
    def interrupt(self):
        move_cursor(len(self.choices) - self.position, "DOWN")
        raise KeyboardInterrupt

    def run(self, default_choice: int = 0):
        "Start the menu and return the selected choice"
        if self.prompt:
            forceWrite(self.prompt, "\n")
        self.position = default_choice
        for i in range(len(self.choices)):
            self.print_choice(i)
            forceWrite("\n")
        move_cursor(len(self.choices) - self.position, "UP")
        with cursor.hide():
            while True:
                choice = self.handle_input()
                if choice is not None:
                    reset_cursor()
                    for _ in range(len(self.choices)):
                        move_cursor(1, "UP")
                        clear_line()
                    forceWrite(f" * {self.choices[choice]}", "\n")
                    return choice
