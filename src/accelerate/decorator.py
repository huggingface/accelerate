from functools import wraps


def on_main_process(func):
    """
    Run func on main process only
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_main_process or not self.use_distributed:
            return func(self, *args, **kwargs)

    return wrapper


def on_local_main_process(func):
    """
    Run func on local main process only
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_local_main_process or not self.use_distributed:
            return func(self, *args, **kwargs)

    return wrapper
