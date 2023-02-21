from ...utils.imports import is_tpu_available


if is_tpu_available(False):
    from torch_xla.distributed import xla_dist

ignored_params = ["help", "positional"]


def add_arguments(argument_group):
    distrib_parser = xla_dist.get_args_parser()
    for action in distrib_parser._actions:
        if action.dest in ignored_params:
            continue
        argument_group._add_action(action)
