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

import functools
import textwrap
import warnings
from typing import Callable, TypeVar

from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


def deprecated(since: str, removed_in: str, instruction: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Marks functions as deprecated.

    It will result in a warning when the function is called and a note in the docstring.

    Args:
        since (`str`):
            The version when the function was first deprecated.
        removed_in (`str`):
            The version when the function will be removed.
        instruction (`str`):
            The action users should take.

    Returns:
        `Callable`: A decorator that will mark the function as deprecated.
    """

    def decorator(function: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            warnings.warn(
                f"'{function.__module__}.{function.__name__}' "
                f"is deprecated in version {since} and will be "
                f"removed in {removed_in}. {instruction}.",
                category=FutureWarning,
                stacklevel=2,
            )
            return function(*args, **kwargs)

        # Add a deprecation note to the docstring.
        docstring = function.__doc__ or ""

        deprecation_note = textwrap.dedent(
            f"""\
            .. deprecated:: {since}
                Deprecated and will be removed in version {removed_in}. {instruction}.
            """
        )

        # Split docstring at first occurrence of newline
        summary_and_body = docstring.split("\n\n", 1)
        if len(summary_and_body) > 1:
            summary, body = summary_and_body
            body = textwrap.dedent(body)
            new_docstring_parts = [deprecation_note, "\n\n", summary, body]
        else:
            summary = summary_and_body[0]
            new_docstring_parts = [deprecation_note, "\n\n", summary]

        wrapper.__doc__ = "".join(new_docstring_parts)

        return wrapper

    return decorator
