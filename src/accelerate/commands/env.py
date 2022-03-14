import argparse
import os
import platform

import numpy as np
import torch

from accelerate import __version__ as version
from accelerate.commands.config import default_config_file, load_config_from_file


def env_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("env")
    else:
        parser = argparse.ArgumentParser("Accelerate env command")

    parser.add_argument(
        "--config_file", default=None, help="The config file to use for the default values in the launching script."
    )

    if subparsers is not None:
        parser.set_defaults(func=env_command)
    return parser


def env_command(args):
    pt_version = torch.__version__
    pt_cuda_available = torch.cuda.is_available()

    accelerate_config = "Not found"
    # Get the default from the config file.
    if args.config_file is not None or os.path.isfile(default_config_file):
        accelerate_config = load_config_from_file(args.config_file).to_dict()

    info = {
        "`Accelerate` version": version,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Numpy version": np.__version__,
        "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
    }

    print("\nCopy-and-paste the text below in your GitHub issue\n")
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]))

    print("- `Accelerate` default config:" if args.config_file is None else "- `Accelerate` config passed:")
    accelerate_config_str = (
        "\n".join([f"\t- {prop}: {val}" for prop, val in accelerate_config.items()])
        if isinstance(accelerate_config, dict)
        else f"\t{accelerate_config}"
    )
    print(accelerate_config_str)

    info["`Accelerate` configs"] = accelerate_config

    return info


def main() -> int:
    parser = env_command_parser()
    args = parser.parse_args()
    env_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
