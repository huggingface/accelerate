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
import argparse


class ArgumentParserWithDashSupport(argparse.ArgumentParser):
    """
    argparse subclass that allows for seamless use of `--a-b` or `--a_b` style arguments automatically.

    Based on the implementation here:
    https://stackoverflow.com/questions/53527387/make-argparse-treat-dashes-and-underscore-identically
    """

    def _parse_optional(self, arg_string):
        # Conditions to not change anything/it's a positional:
        # - Empty string, positional
        # - Doesn't start with a prefix
        # - Single character
        if (not arg_string) or (arg_string[0] not in self.prefix_chars) or (len(arg_string) == 1):
            return None

        option_tuples = self._get_option_tuples(arg_string)

        # If multiple matches, it was ambigous, raise an error
        if len(option_tuples) > 1:
            options = ", ".join([option_string for _, option_string, _ in option_tuples])
            self.error(f"ambiguous option: {arg_string} could match {options}")

        # If exactly one match, return it
        elif len(option_tuples) == 1:
            (option_tuple,) = option_tuples
            return option_tuple

        # If not found, but looks like a negative number, probably posisional
        if self._negative_number_matcher.match(arg_string) and not self._has_negative_number_optionals:
            return None

        # If it has a space, probably positional
        if " " in arg_string:
            return None

        # Otherwise meant to be optional though no such option,
        # but could be valid in a subparser so just return it
        return None, arg_string, None

    def _get_option_tuples(self, option_string):
        result = []
        explicit_arg = None
        if "=" in option_string:
            option_prefix, explicit_arg = option_string.split("=", 1)
        else:
            option_prefix = option_string
        # Assuming it's a perfect match
        if option_prefix in self._option_string_actions:
            action = self._option_string_actions[option_prefix]
            result.append((action, option_prefix, explicit_arg))
        else:
            # Imperfect match, have to go dig
            chars = self.prefix_chars
            if option_string[0] in chars and option_string[1] not in chars:
                # short option: if single character, can be concatenated with arguments
                short_option_prefix = option_string[:2]
                short_explicit_arg = option_string[2:]
                if short_option_prefix in self._option_string_actions:
                    action = self._option_string_actions[short_option_prefix]
                    result.append((action, short_option_prefix, short_explicit_arg))

            # Finally check for `-` vs `_`
            underscored = {k.replace("-", "_"): k for k in self._option_string_actions}
            option_prefix = option_prefix.replace("-", "_")
            if option_prefix in underscored:
                action = self._option_string_actions[underscored[option_prefix]]
                result.append((action, underscored[option_prefix], explicit_arg))
            elif self.allow_abbrev:
                for option_string in underscored:
                    if option_string.startswith(option_prefix):
                        action = self._option_string_actions[underscored[option_string]]
                        result.append((action, underscored[option_string], explicit_arg))
        return result
