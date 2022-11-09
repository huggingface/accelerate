# Copyright 2022 The HuggingFace Team and Brian Chao. All rights reserved.
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
Utilities relating to parsing raw characters from the keyboard, based on https://github.com/bchao1/bullet
"""


import string
import sys
import termios
import tty


TAB_KEY = ord("\t")
NEWLINE_KEY = ord("\r")
ESC_KEY = 27
ARROW_KEY_FLAG = 1 << 8
ARROW_UP_KEY = 65 + ARROW_KEY_FLAG
ARROW_DOWN_KEY = 66 + ARROW_KEY_FLAG
ARROW_RIGHT_KEY = 67 + ARROW_KEY_FLAG
ARROW_LEFT_KEY = 68 + ARROW_KEY_FLAG
ARROW_KEY_BEGIN = ARROW_UP_KEY
ARROW_KEY_END = ARROW_LEFT_KEY
MOD_KEY_INT = 91
UNDEFINED_KEY = sys.maxsize
INTERRUPT_KEY = 3


def get_raw_chars():
    "Gets raw characters from inputs"
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def get_character():
    "Gets a character from the keyboard and returns the key code"
    char = get_raw_chars()
    if ord(char) in [INTERRUPT_KEY, NEWLINE_KEY]:
        return char

    elif ord(char) == ESC_KEY:
        combo = get_raw_chars()
        if ord(combo) == MOD_KEY_INT:
            key = get_raw_chars()
            if ord(key) >= ARROW_KEY_BEGIN - ARROW_KEY_FLAG and ord(key) <= ARROW_KEY_END - ARROW_KEY_FLAG:
                return chr(ord(key) + ARROW_KEY_FLAG)
            else:
                return UNDEFINED_KEY
        else:
            return get_raw_chars()

    else:
        if char in string.printable:
            return char
        else:
            return UNDEFINED_KEY
