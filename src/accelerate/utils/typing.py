from typing import Callable, Optional, ParamSpec, TypeVar


T = TypeVar("T")
P = ParamSpec("P")


def take_annotation_from(this: Callable[P, Optional[T]]) -> Callable[[Callable], Callable[P, Optional[T]]]:
    def decorator(real_function: Callable) -> Callable[P, Optional[T]]:
        def new_function(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            return real_function(*args, **kwargs)

        return new_function

    return decorator
