from ...utils.imports import is_tpu_available


if is_tpu_available(False):
    from torch_xla.distributed.xla_dist import get_args_parser

ignored_params = ["help", "positional"]


def add_arguments(argument_group):
    distrib_parser = get_args_parser()
    for action in distrib_parser._actions:
        if action.dest in ignored_params:
            continue
        argument_group._add_action(action)
